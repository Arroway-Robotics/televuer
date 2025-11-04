from vuer import Vuer
from vuer.schemas import ImageBackground, Hands, MotionControllers, WebRTCVideoPlane, WebRTCStereoVideoPlane
from multiprocessing import Value, Array, Process, shared_memory
import numpy as np
import asyncio
import cv2
import os
from pathlib import Path



import asyncio
import threading
import time
import socket
import cv2
from fastapi import FastAPI
from starlette.responses import StreamingResponse, Response
import uvicorn

# One-time server state holders (attach to self if inside a class)
_app = None
_server_thread = None
_encoder_thread = None
_latest_jpeg = None
_latest_lock = threading.Lock()

class TeleVuer:
    def __init__(self, binocular: bool, use_hand_tracking: bool, img_shape, img_shm_name, cert_file=None, key_file=None, ngrok=False, webrtc=False):
        """
        TeleVuer class for OpenXR-based XR teleoperate applications.
        This class handles the communication with the Vuer server and manages the shared memory for image and pose data.

        :param binocular: bool, whether the application is binocular (stereoscopic) or monocular.
        :param use_hand_tracking: bool, whether to use hand tracking or controller tracking.
        :param img_shape: tuple, shape of the image (height, width, channels).
        :param img_shm_name: str, name of the shared memory for the image.
        :param cert_file: str, path to the SSL certificate file.
        :param key_file: str, path to the SSL key file.
        :param ngrok: bool, whether to use ngrok for tunneling.
        """
        self.binocular = binocular
        self.use_hand_tracking = use_hand_tracking
        self.img_height = img_shape[0]
        if self.binocular:
            self.img_width  = img_shape[1] // 2
        else:
            self.img_width  = img_shape[1]
        
        current_module_dir = Path(__file__).resolve().parent.parent.parent
        self.cert_file = cert_file
        self.key_file = key_file
        if cert_file is None:
            self.cert_file = os.path.join(current_module_dir, "cert.pem")
        if key_file is None:
            self.key_file = os.path.join(current_module_dir, "key.pem")

        if ngrok:
            self.vuer = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)
        else:
            self.vuer = Vuer(host='0.0.0.0', cert=self.cert_file, key=self.key_file, queries=dict(grid=False), queue_len=3)

        self.vuer.add_handler("CAMERA_MOVE")(self.on_cam_move)
        if self.use_hand_tracking:
            self.vuer.add_handler("HAND_MOVE")(self.on_hand_move)
        else:
            self.vuer.add_handler("CONTROLLER_MOVE")(self.on_controller_move)

        existing_shm = shared_memory.SharedMemory(name=img_shm_name)
        self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=existing_shm.buf)

        self.webrtc = webrtc
        if self.binocular and not self.webrtc:
            self.vuer.spawn(start=False)(self.main_image_binocular)
        elif not self.binocular and not self.webrtc:
            self.vuer.spawn(start=False)(self.main_image_monocular)
        elif self.webrtc:
            self.vuer.spawn(start=False)(self.main_image_mjpeg)

        self.head_pose_shared = Array('d', 16, lock=True)
        self.left_arm_pose_shared = Array('d', 16, lock=True)
        self.right_arm_pose_shared = Array('d', 16, lock=True)
        if self.use_hand_tracking:
            self.left_hand_position_shared = Array('d', 75, lock=True)
            self.right_hand_position_shared = Array('d', 75, lock=True)
            self.left_hand_orientation_shared = Array('d', 25 * 9, lock=True)
            self.right_hand_orientation_shared = Array('d', 25 * 9, lock=True)

            self.left_pinch_state_shared = Value('b', False, lock=True)
            self.left_pinch_value_shared = Value('d', 0.0, lock=True)
            self.left_squeeze_state_shared = Value('b', False, lock=True)
            self.left_squeeze_value_shared = Value('d', 0.0, lock=True)

            self.right_pinch_state_shared = Value('b', False, lock=True)
            self.right_pinch_value_shared = Value('d', 0.0, lock=True)
            self.right_squeeze_state_shared = Value('b', False, lock=True)
            self.right_squeeze_value_shared = Value('d', 0.0, lock=True)
        else:
            self.left_trigger_state_shared = Value('b', False, lock=True)
            self.left_trigger_value_shared = Value('d', 0.0, lock=True)
            self.left_squeeze_state_shared = Value('b', False, lock=True)
            self.left_squeeze_value_shared = Value('d', 0.0, lock=True)
            self.left_thumbstick_state_shared = Value('b', False, lock=True)
            self.left_thumbstick_value_shared = Array('d', 2, lock=True)
            self.left_aButton_shared = Value('b', False, lock=True)
            self.left_bButton_shared = Value('b', False, lock=True)

            self.right_trigger_state_shared = Value('b', False, lock=True)
            self.right_trigger_value_shared = Value('d', 0.0, lock=True)
            self.right_squeeze_state_shared = Value('b', False, lock=True)
            self.right_squeeze_value_shared = Value('d', 0.0, lock=True)
            self.right_thumbstick_state_shared = Value('b', False, lock=True)
            self.right_thumbstick_value_shared = Array('d', 2, lock=True)
            self.right_aButton_shared = Value('b', False, lock=True)
            self.right_bButton_shared = Value('b', False, lock=True)

        self._mjpeg_app = None
        self._mjpeg_server_thread = None
        self._mjpeg_capture_thread = None  # Independent RealSense capture
        self._mjpeg_encoder_thread = None
        self._mjpeg_latest_frame = None  # Independent frame buffer
        self._mjpeg_latest_jpeg = None
        self._mjpeg_lock = threading.Lock()
        self._mjpeg_port = 8765  # default; change if you want

        self.process = Process(target=self.vuer_run)
        self.process.daemon = True
        self.process.start()



    def vuer_run(self):
        self.vuer.run()

    async def on_cam_move(self, event, session, fps=60):
        try:
            with self.head_pose_shared.get_lock():
                self.head_pose_shared[:] = event.value["camera"]["matrix"]
        except:
            pass

    async def on_controller_move(self, event, session, fps=60):
        try:
            with self.left_arm_pose_shared.get_lock():
                self.left_arm_pose_shared[:] = event.value["left"]
            with self.right_arm_pose_shared.get_lock():
                self.right_arm_pose_shared[:] = event.value["right"]

            left_controller_state = event.value["leftState"]
            right_controller_state = event.value["rightState"]

            def extract_controller_states(state_dict, prefix):
                # trigger
                with getattr(self, f"{prefix}_trigger_state_shared").get_lock():
                    getattr(self, f"{prefix}_trigger_state_shared").value = bool(state_dict.get("trigger", False))
                with getattr(self, f"{prefix}_trigger_value_shared").get_lock():
                    getattr(self, f"{prefix}_trigger_value_shared").value = float(state_dict.get("triggerValue", 0.0))
                # squeeze
                with getattr(self, f"{prefix}_squeeze_state_shared").get_lock():
                    getattr(self, f"{prefix}_squeeze_state_shared").value = bool(state_dict.get("squeeze", False))
                with getattr(self, f"{prefix}_squeeze_value_shared").get_lock():
                    getattr(self, f"{prefix}_squeeze_value_shared").value = float(state_dict.get("squeezeValue", 0.0))
                # thumbstick
                with getattr(self, f"{prefix}_thumbstick_state_shared").get_lock():
                    getattr(self, f"{prefix}_thumbstick_state_shared").value = bool(state_dict.get("thumbstick", False))
                with getattr(self, f"{prefix}_thumbstick_value_shared").get_lock():
                    getattr(self, f"{prefix}_thumbstick_value_shared")[:] = state_dict.get("thumbstickValue", [0.0, 0.0])
                # buttons
                with getattr(self, f"{prefix}_aButton_shared").get_lock():
                    getattr(self, f"{prefix}_aButton_shared").value = bool(state_dict.get("aButton", False))
                with getattr(self, f"{prefix}_bButton_shared").get_lock():
                    getattr(self, f"{prefix}_bButton_shared").value = bool(state_dict.get("bButton", False))

            extract_controller_states(left_controller_state, "left")
            extract_controller_states(right_controller_state, "right")
        except:
            pass

    async def on_hand_move(self, event, session, fps=60):
        try:
            left_hand_data = event.value["left"]
            right_hand_data = event.value["right"]
            left_hand_state = event.value["leftState"]
            right_hand_state = event.value["rightState"]

            def extract_hand_poses(hand_data, arm_pose_shared, hand_position_shared, hand_orientation_shared):
                with arm_pose_shared.get_lock():
                    arm_pose_shared[:] = hand_data[0:16]

                with hand_position_shared.get_lock():
                    for i in range(25):
                        base = i * 16
                        hand_position_shared[i * 3: i * 3 + 3] = [hand_data[base + 12], hand_data[base + 13], hand_data[base + 14]]

                with hand_orientation_shared.get_lock():
                    for i in range(25):
                        base = i * 16
                        hand_orientation_shared[i * 9: i * 9 + 9] = [
                            hand_data[base + 0], hand_data[base + 1], hand_data[base + 2],
                            hand_data[base + 4], hand_data[base + 5], hand_data[base + 6],
                            hand_data[base + 8], hand_data[base + 9], hand_data[base + 10],
                        ]

            def extract_hand_states(state_dict, prefix):
                # pinch
                with getattr(self, f"{prefix}_pinch_state_shared").get_lock():
                    getattr(self, f"{prefix}_pinch_state_shared").value = bool(state_dict.get("pinch", False))
                with getattr(self, f"{prefix}_pinch_value_shared").get_lock():
                    getattr(self, f"{prefix}_pinch_value_shared").value = float(state_dict.get("pinchValue", 0.0))
                # squeeze
                with getattr(self, f"{prefix}_squeeze_state_shared").get_lock():
                    getattr(self, f"{prefix}_squeeze_state_shared").value = bool(state_dict.get("squeeze", False))
                with getattr(self, f"{prefix}_squeeze_value_shared").get_lock():
                    getattr(self, f"{prefix}_squeeze_value_shared").value = float(state_dict.get("squeezeValue", 0.0))

            extract_hand_poses(left_hand_data, self.left_arm_pose_shared, self.left_hand_position_shared, self.left_hand_orientation_shared)
            extract_hand_poses(right_hand_data, self.right_arm_pose_shared, self.right_hand_position_shared, self.right_hand_orientation_shared)
            extract_hand_states(left_hand_state, "left")
            extract_hand_states(right_hand_state, "right")

        except:
            pass
    
    async def main_image_binocular(self, session, fps=60):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        # session.set @ Scene()

        while True:
            display_image = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2RGB)
            # aspect_ratio = self.img_width / self.img_height
            session.upsert(
                [
                    ImageBackground(
                        display_image[:, :self.img_width],
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        # The underlying rendering engine supported a layer binary bitmask for both objects and the camera. 
                        # Below we set the two image planes, left and right, to layers=1 and layers=2. 
                        # Note that these two masks are associated with left eye’s camera and the right eye’s camera.
                        layers=1,
                        format="jpeg",
                        quality=100,
                        key="background-left",
                        interpolate=True,
                    ),
                    ImageBackground(
                        display_image[:, self.img_width:],
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        layers=2,
                        format="jpeg",
                        quality=100,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            # 'jpeg' encoding should give you about 30fps with a 16ms wait in-between.
            await asyncio.sleep(0.016 * 2)

    async def main_image_monocular(self, session, fps=60):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            display_image = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2RGB)
            # aspect_ratio = self.img_width / self.img_height
            session.upsert(
                [
                    ImageBackground(
                        display_image,
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        format="jpeg",
                        quality=50,
                        key="background-mono",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            await asyncio.sleep(0.016)

    async def main_image_mjpeg(self, session, fps=60):
        """
        Display robot camera feed using MJPEG streaming via FastAPI.
        This is simpler and more reliable than direct WebRTC decoding.
        """
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    showLeft=False,
                    showRight=False
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    showLeft=False,
                    showRight=False,
                )
            )
    
        """
        Display robot camera feed using MJPEG (FastAPI + Uvicorn).
        One-time embed in Vuer; browser pulls frames directly.
        """
        # 0) Input/UI widgets
        if self.use_hand_tracking:
            session.upsert(Hands(stream=True, key="hands", showLeft=False, showRight=False), to="bgChildren")
        else:
            session.upsert(MotionControllers(stream=True, key="motionControllers", showLeft=False, showRight=False))

        # 1) Lazy start the MJPEG server
        import asyncio, time, socket, cv2, uvicorn
        from fastapi import FastAPI
        from starlette.responses import StreamingResponse, Response

        port = None
        jpeg_quality=70
        force_https=None  # None=auto-detect, True=force HTTPS, False=force HTTP

        if port is None:
            port = self._mjpeg_port

        def _get_lan_ip() -> str:
            ip = "127.0.0.1"
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                s.close()
            except Exception:
                pass
            return ip

        if self._mjpeg_app is None:
            app = FastAPI()
            
            # Add CORS middleware to allow cross-origin requests
            from fastapi.middleware.cors import CORSMiddleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],  # Allow all origins
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            boundary = b"frame"

            @app.get("/mjpeg")
            async def mjpeg():
                async def gen():
                    while True:
                        with self._mjpeg_lock:
                            blob = self._mjpeg_latest_jpeg
                        if blob:
                            yield b"--" + boundary + b"\r\n"
                            yield b"Content-Type: image/jpeg\r\n"
                            yield f"Content-Length: {len(blob)}\r\n\r\n".encode()
                            yield blob + b"\r\n"
                        await asyncio.sleep(0.001)
                return StreamingResponse(
                    gen(),
                    media_type="multipart/x-mixed-replace; boundary=frame",
                    headers={
                        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                        "Pragma": "no-cache",
                        "Connection": "close",
                    },
                )

            @app.get("/frame.jpg")
            def snapshot():
                with self._mjpeg_lock:
                    blob = self._mjpeg_latest_jpeg
                return Response(blob or b"", media_type="image/jpeg")

            self._mjpeg_app = app

            # HTTPS if certs are present AND force_https is not False
            ssl_kwargs = {}
            if force_https is None:
                # Auto-detect: use HTTPS if certs are available
                use_https = bool(self.cert_file and self.key_file)
            else:
                # Explicit override
                use_https = bool(force_https)
            
            if use_https:
                if self.cert_file and self.key_file:
                    ssl_kwargs = {"ssl_certfile": self.cert_file, "ssl_keyfile": self.key_file}
                else:
                    print("[WARN] HTTPS requested but no certificates available, falling back to HTTP")
                    use_https = False

            cfg = uvicorn.Config(self._mjpeg_app, host="0.0.0.0", port=port, log_level="warning", **ssl_kwargs)
            server = uvicorn.Server(cfg)

            # Start Uvicorn in a background thread
            self._mjpeg_server_thread = threading.Thread(target=server.run, daemon=True)
            self._mjpeg_server_thread.start()

        # 2) Start independent RealSense capture for MJPEG (doesn't use self.img_array)
        if self._mjpeg_capture_thread is None:
            import pyrealsense2 as rs
            
            def _capture_loop():
                """Independent RealSense capture for MJPEG (no conflict with self.img_array)"""
                try:
                    # Initialize RealSense pipeline
                    pipeline = rs.pipeline()
                    config = rs.config()
                    
                    # Configure for color stream only (faster than depth+color)
                    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
                    
                    print("[INFO] Starting independent RealSense capture for MJPEG...")
                    pipeline.start(config)
                    print("[INFO] RealSense capture started")
                    
                    while True:
                        try:
                            # Wait for frames (with timeout)
                            frames = pipeline.wait_for_frames(timeout_ms=1000)
                            color_frame = frames.get_color_frame()
                            
                            if color_frame:
                                # Convert to numpy array
                                frame_bgr = np.asanyarray(color_frame.get_data())
                                
                                # Store in MJPEG frame buffer (not self.img_array)
                                with self._mjpeg_lock:
                                    self._mjpeg_latest_frame = frame_bgr.copy()
                        
                        except Exception as e:
                            print(f"[WARN] RealSense frame capture error: {e}")
                            time.sleep(0.01)
                            
                except Exception as e:
                    print(f"[ERROR] Failed to initialize RealSense for MJPEG: {e}")
                    print(f"[INFO] Falling back to self.img_array (may cause conflicts)")
                    # Fallback: use self.img_array if RealSense init fails
                    while True:
                        try:
                            if self.img_array is not None:
                                with self._mjpeg_lock:
                                    self._mjpeg_latest_frame = self.img_array.copy()
                        except Exception:
                            pass
                        time.sleep(0.033)  # ~30fps
                finally:
                    try:
                        pipeline.stop()
                    except:
                        pass
            
            self._mjpeg_capture_thread = threading.Thread(target=_capture_loop, daemon=True)
            self._mjpeg_capture_thread.start()
        
        # 3) Start the encoder thread once (encodes from _mjpeg_latest_frame)
        if self._mjpeg_encoder_thread is None:
            def _encoder_loop():
                period = 1.0 / max(1, fps)
                while True:
                    frame_bgr = None
                    try:
                        # Read from independent MJPEG frame buffer (not self.img_array)
                        with self._mjpeg_lock:
                            if hasattr(self, '_mjpeg_latest_frame') and self._mjpeg_latest_frame is not None:
                                frame_bgr = self._mjpeg_latest_frame
                    except Exception:
                        pass

                    if frame_bgr is not None:
                        ok, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                        if ok:
                            with self._mjpeg_lock:
                                self._mjpeg_latest_jpeg = jpg.tobytes()
                    time.sleep(period)

            self._mjpeg_encoder_thread = threading.Thread(target=_encoder_loop, daemon=True)
            self._mjpeg_encoder_thread.start()

        # 3) One-time embed in Vuer
        # Wait a bit for server to start
        await asyncio.sleep(2)
        
        # Determine scheme based on force_https setting
        if force_https is None:
            scheme = "https" if (self.cert_file and self.key_file) else "http"
        else:
            scheme = "https" if force_https else "http"
        lan_ip = _get_lan_ip()
        stream_url = f"{scheme}://{lan_ip}:{port}/mjpeg"
        
        # Debug: Check server and frame status
        print(f"[DEBUG] MJPEG Configuration:")
        print(f"  - force_https: {force_https}")
        print(f"  - Scheme: {scheme}")
        print(f"  - LAN IP: {lan_ip}")
        print(f"  - Port: {port}")
        print(f"  - URL: {stream_url}")
        print(f"  - Has certs: {bool(self.cert_file and self.key_file)}")
        print(f"  - Using HTTPS: {scheme == 'https'}")
        
        # Check if we have frames
        try:
            frame_shape = self.img_array.shape if self.img_array is not None else None
            print(f"  - Frame shape: {frame_shape}")
        except Exception as e:
            print(f"  - Frame error: {e}")
        
        # Check if JPEG is being encoded
        with self._mjpeg_lock:
            jpeg_size = len(self._mjpeg_latest_jpeg) if self._mjpeg_latest_jpeg else 0
        print(f"  - JPEG size: {jpeg_size} bytes")
        
        # Test server accessibility
        try:
            import requests
            test_url = f"{scheme}://localhost:{port}/frame.jpg"
            # Disable SSL verification for self-signed certs
            response = requests.get(test_url, timeout=2, verify=False)
            print(f"  - Server test: {response.status_code}, {len(response.content)} bytes")
        except Exception as e:
            print(f"  - Server test failed: {e}")

        # Display using independent MJPEG frame buffer (not self.img_array)
        from vuer.schemas import ImageBackground
        import numpy as np
        
        print(f"[INFO] Starting optimized video display (low FPS to reduce VR lag)")
        print(f"[INFO] Using independent RealSense capture (no conflict with self.img_array)")
        print(f"[INFO] MJPEG stream available for external viewing at: {stream_url}")
        
        # Low frame rate to minimize VR lag
        target_fps = 15  # Much lower than main_image() to reduce lag
        frame_interval = 1.0 / target_fps
        
        frame_count = 0
        
        while True:
            try:
                # Read from independent MJPEG frame buffer (not self.img_array)
                frame_bgr = None
                with self._mjpeg_lock:
                    if hasattr(self, '_mjpeg_latest_frame') and self._mjpeg_latest_frame is not None:
                        frame_bgr = self._mjpeg_latest_frame.copy()
                
                if frame_bgr is not None:
                    # Convert BGR to RGB
                    display_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    
                    # Update Vuer ImageBackground
                    session.upsert(
                        [
                            ImageBackground(
                                display_image,
                                aspect=1.778,
                                height=1,
                                distanceToCamera=1,
                                format="jpeg",
                                quality=60,  # Lower quality for faster encoding
                                key="mjpeg-bg",
                                interpolate=False,
                            )
                        ],
                        to="bgChildren",
                    )
                    
                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"[INFO] Updated {frame_count} frames to Vuer ({target_fps} fps)")
                
                # Sleep to maintain target FPS
                await asyncio.sleep(frame_interval)
                
            except Exception as e:
                print(f"[WARN] Frame update error: {e}")
                await asyncio.sleep(0.1)




    # ==================== common data ====================
    @property
    def head_pose(self):
        """np.ndarray, shape (4, 4), head SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.head_pose_shared.get_lock():
            return np.array(self.head_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def left_arm_pose(self):
        """np.ndarray, shape (4, 4), left arm SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.left_arm_pose_shared.get_lock():
            return np.array(self.left_arm_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def right_arm_pose(self):
        """np.ndarray, shape (4, 4), right arm SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.right_arm_pose_shared.get_lock():
            return np.array(self.right_arm_pose_shared[:]).reshape(4, 4, order="F")

    # ==================== Hand Tracking Data ====================
    @property
    def left_hand_positions(self):
        """np.ndarray, shape (25, 3), left hand 25 landmarks' 3D positions."""
        with self.left_hand_position_shared.get_lock():
            return np.array(self.left_hand_position_shared[:]).reshape(25, 3)

    @property
    def right_hand_positions(self):
        """np.ndarray, shape (25, 3), right hand 25 landmarks' 3D positions."""
        with self.right_hand_position_shared.get_lock():
            return np.array(self.right_hand_position_shared[:]).reshape(25, 3)

    @property
    def left_hand_orientations(self):
        """np.ndarray, shape (25, 3, 3), left hand 25 landmarks' orientations (flattened 3x3 matrices, column-major)."""
        with self.left_hand_orientation_shared.get_lock():
            return np.array(self.left_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order="F")

    @property
    def right_hand_orientations(self):
        """np.ndarray, shape (25, 3, 3), right hand 25 landmarks' orientations (flattened 3x3 matrices, column-major)."""
        with self.right_hand_orientation_shared.get_lock():
            return np.array(self.right_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order="F")

    @property
    def left_hand_pinch_state(self):
        """bool, whether left hand is pinching."""
        with self.left_pinch_state_shared.get_lock():
            return self.left_pinch_state_shared.value

    @property
    def left_hand_pinch_value(self):
        """float, pinch strength of left hand."""
        with self.left_pinch_value_shared.get_lock():
            return self.left_pinch_value_shared.value

    @property
    def left_hand_squeeze_state(self):
        """bool, whether left hand is squeezing."""
        with self.left_squeeze_state_shared.get_lock():
            return self.left_squeeze_state_shared.value

    @property
    def left_hand_squeeze_value(self):
        """float, squeeze strength of left hand."""
        with self.left_squeeze_value_shared.get_lock():
            return self.left_squeeze_value_shared.value

    @property
    def right_hand_pinch_state(self):
        """bool, whether right hand is pinching."""
        with self.right_pinch_state_shared.get_lock():
            return self.right_pinch_state_shared.value

    @property
    def right_hand_pinch_value(self):
        """float, pinch strength of right hand."""
        with self.right_pinch_value_shared.get_lock():
            return self.right_pinch_value_shared.value

    @property
    def right_hand_squeeze_state(self):
        """bool, whether right hand is squeezing."""
        with self.right_squeeze_state_shared.get_lock():
            return self.right_squeeze_state_shared.value

    @property
    def right_hand_squeeze_value(self):
        """float, squeeze strength of right hand."""
        with self.right_squeeze_value_shared.get_lock():
            return self.right_squeeze_value_shared.value

    # ==================== Controller Data ====================
    @property
    def left_controller_trigger_state(self):
        """bool, left controller trigger pressed or not."""
        with self.left_trigger_state_shared.get_lock():
            return self.left_trigger_state_shared.value

    @property
    def left_controller_trigger_value(self):
        """float, left controller trigger analog value (0.0 ~ 1.0)."""
        with self.left_trigger_value_shared.get_lock():
            return self.left_trigger_value_shared.value

    @property
    def left_controller_squeeze_state(self):
        """bool, left controller squeeze pressed or not."""
        with self.left_squeeze_state_shared.get_lock():
            return self.left_squeeze_state_shared.value

    @property
    def left_controller_squeeze_value(self):
        """float, left controller squeeze analog value (0.0 ~ 1.0)."""
        with self.left_squeeze_value_shared.get_lock():
            return self.left_squeeze_value_shared.value

    @property
    def left_controller_thumbstick_state(self):
        """bool, whether left thumbstick is touched or clicked."""
        with self.left_thumbstick_state_shared.get_lock():
            return self.left_thumbstick_state_shared.value

    @property
    def left_controller_thumbstick_value(self):
        """np.ndarray, shape (2,), left thumbstick 2D axis values (x, y)."""
        with self.left_thumbstick_value_shared.get_lock():
            return np.array(self.left_thumbstick_value_shared[:])

    @property
    def left_controller_aButton(self):
        """bool, left controller 'A' button pressed."""
        with self.left_aButton_shared.get_lock():
            return self.left_aButton_shared.value

    @property
    def left_controller_bButton(self):
        """bool, left controller 'B' button pressed."""
        with self.left_bButton_shared.get_lock():
            return self.left_bButton_shared.value

    @property
    def right_controller_trigger_state(self):
        """bool, right controller trigger pressed or not."""
        with self.right_trigger_state_shared.get_lock():
            return self.right_trigger_state_shared.value

    @property
    def right_controller_trigger_value(self):
        """float, right controller trigger analog value (0.0 ~ 1.0)."""
        with self.right_trigger_value_shared.get_lock():
            return self.right_trigger_value_shared.value

    @property
    def right_controller_squeeze_state(self):
        """bool, right controller squeeze pressed or not."""
        with self.right_squeeze_state_shared.get_lock():
            return self.right_squeeze_state_shared.value

    @property
    def right_controller_squeeze_value(self):
        """float, right controller squeeze analog value (0.0 ~ 1.0)."""
        with self.right_squeeze_value_shared.get_lock():
            return self.right_squeeze_value_shared.value

    @property
    def right_controller_thumbstick_state(self):
        """bool, whether right thumbstick is touched or clicked."""
        with self.right_thumbstick_state_shared.get_lock():
            return self.right_thumbstick_state_shared.value

    @property
    def right_controller_thumbstick_value(self):
        """np.ndarray, shape (2,), right thumbstick 2D axis values (x, y)."""
        with self.right_thumbstick_value_shared.get_lock():
            return np.array(self.right_thumbstick_value_shared[:])

    @property
    def right_controller_aButton(self):
        """bool, right controller 'A' button pressed."""
        with self.right_aButton_shared.get_lock():
            return self.right_aButton_shared.value

    @property
    def right_controller_bButton(self):
        """bool, right controller 'B' button pressed."""
        with self.right_bButton_shared.get_lock():
            return self.right_bButton_shared.value