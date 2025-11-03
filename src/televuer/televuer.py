from vuer import Vuer
from vuer.schemas import ImageBackground, Hands, MotionControllers, WebRTCVideoPlane, WebRTCStereoVideoPlane
from multiprocessing import Value, Array, Process, shared_memory
import numpy as np
import asyncio
import cv2
import os
from pathlib import Path


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
        if cert_file is None:
            cert_file = os.path.join(current_module_dir, "cert.pem")
        if key_file is None:
            key_file = os.path.join(current_module_dir, "key.pem")

        if ngrok:
            self.vuer = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)
        else:
            self.vuer = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)

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
            self.vuer.spawn(start=False)(self.main_image_webrtc)

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
    


        # Create an OpenCV window and display a blank image
        # height, width = 720, 1280  # Adjust the size as needed
        # img = np.zeros((height, width, 3), dtype=np.uint8)
        # cv2.imshow('Video', img)
        # cv2.waitKey(1)  # Ensure the window is created

        import asyncio
        import logging
        import threading
        import time
        import cv2
        from fastapi import FastAPI, Response
        import uvicorn
        from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
        from aiortc import MediaStreamTrack
        import av
        from vuer.schemas import Html
        
        # Suppress FFmpeg/libav H.264 decoder warnings
        av.logging.set_level(av.logging.FATAL)
        
        # Shared state for MJPEG streaming
        last_jpeg = b""
        last_frame = None
        frame_lock = threading.Lock()
        
        # FastAPI app for MJPEG streaming
        app = FastAPI()
        
        @app.get("/mjpeg")
        def mjpeg():
            def gen():
                nonlocal last_jpeg
                boundary = b"--frame"
                while True:
                    with frame_lock:
                        jpg = last_jpeg
                    if not jpg:
                        time.sleep(0.005)
                        continue
                    yield (boundary + b"\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                           jpg + b"\r\n")
            return Response(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
        
        # JPEG encoding thread
        def jpeg_encoder_loop():
            nonlocal last_jpeg
            enc_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            while True:
                with frame_lock:
                    frame = last_frame
                if frame is None:
                    time.sleep(0.002)
                    continue
                ok, buf = cv2.imencode(".jpg", frame, enc_params)
                if ok:
                    with frame_lock:
                        last_jpeg = buf.tobytes() 

        # WebRTC frame receiver
        async def recv_camera_stream(track: MediaStreamTrack):
            nonlocal last_frame
            print("[INFO] Starting video frame reception...")
            frame_count = 0
            while True:
                try:
                    frame = await track.recv()
                    # Convert to BGR numpy array
                    img = frame.to_ndarray(format="bgr24")
                    with frame_lock:
                        last_frame = img
                    frame_count += 1
                    if frame_count % 100 == 0:
                        print(f"[INFO] Received {frame_count} frames")
                except Exception as e:
                    if frame_count < 10:
                        print(f"[WARN] Frame decode error: {e}")
                    continue
        
        # WebRTC connection setup
        def run_webrtc_loop(loop):
            asyncio.set_event_loop(loop)
            async def setup():
                try:
                    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.123.161")
                    print("[INFO] Connecting to robot...")
                    await conn.connect()
                    print("[INFO] Connected successfully")
                    
                    conn.video.switchVideoChannel(True)
                    print("[INFO] Video channel enabled")
                    
                    conn.video.add_track_callback(recv_camera_stream)
                    print("[INFO] Video callback registered")
                except Exception as e:
                    print(f"[ERROR] WebRTC connection failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            loop.run_until_complete(setup())
            loop.run_forever()
        
        # Start JPEG encoder thread
        print("[INFO] Starting JPEG encoder thread...")
        jpeg_thread = threading.Thread(target=jpeg_encoder_loop, daemon=True)
        jpeg_thread.start()
        
        # Start WebRTC thread
        print("[INFO] Starting WebRTC thread...")
        webrtc_loop = asyncio.new_event_loop()
        webrtc_thread = threading.Thread(target=run_webrtc_loop, args=(webrtc_loop,), daemon=True)
        webrtc_thread.start()
        
        # Start FastAPI server in a separate thread
        print("[INFO] Starting MJPEG server on port 8765...")
        def run_server():
            uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        await asyncio.sleep(2)
        
        # Get the host IP
        import socket
        hostname = socket.gethostname()
        host_ip = socket.gethostbyname(hostname)
        
        print(f"[INFO] MJPEG stream available at http://{host_ip}:8765/mjpeg")
        
        # Insert HTML img tag for MJPEG stream
        session.upsert(
            Html(f"""
              <img src="http://{host_ip}:8765/mjpeg"
                   style="width:100%; height:100%; object-fit:contain;" />
            """),
            key="mjpeg-bg",
            to="bgChildren",
        )
        
        print("[INFO] MJPEG video display configured")
        
        # Keep the async function running
        try:
            while True:
                await asyncio.sleep(1)
        finally:
            print("[INFO] Shutting down...")


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