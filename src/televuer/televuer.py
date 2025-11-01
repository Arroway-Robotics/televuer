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

    async def main_image_webrtc(self, session, fps=60):
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
        from queue import Queue
        from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
        from aiortc import MediaStreamTrack
        from collections import deque
        import av
        
        # Suppress FFmpeg/libav H.264 decoder warnings at the C library level
        # These warnings are expected during WebRTC initialization and are handled gracefully
        av.logging.set_level(av.logging.FATAL)  # Only show fatal errors, suppress warnings

        frame_queue = Queue()
        # frame_queue = deque(maxlen=1) 

        # Choose a connection method (uncomment the correct one)
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.123.161")
        # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="10.1.10.48")

        # Async function to receive video frames and put them in the queue
        async def recv_camera_stream(track: MediaStreamTrack):
            # Note: go2_webrtc_driver already handles the first frame in on_track handler
            # So we don't need to discard it here
            frame_count = 0
            while True:
                try:
                    frame = await track.recv()
                    # Convert the frame to a NumPy array
                    img = frame.to_ndarray(format="bgr24")
                    frame_queue.put(img)
                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"[DEBUG] Successfully received and decoded {frame_count} frames, queue size: {frame_queue.qsize()}")
                except Exception as e:
                    print(f"[DEBUG] Frame decode error: {e}")
                    continue

        def run_asyncio_loop(loop):
            asyncio.set_event_loop(loop)
            async def setup():
                try:
                    # Connect to the device
                    print("[DEBUG] Connecting to robot...")
                    await conn.connect()
                    print("[DEBUG] Connected successfully")

                    # Switch video channel on and start receiving video frames
                    print("[DEBUG] Switching video channel ON")
                    conn.video.switchVideoChannel(True)
                    print(f"[DEBUG] Video channel switched, video object: {conn.video}")
                    print(f"[DEBUG] Track callbacks list: {conn.video.track_callbacks}")

                    # Add callback to handle received video frames
                    print("[DEBUG] Adding track callback...")
                    conn.video.add_track_callback(recv_camera_stream)
                    print(f"[DEBUG] Callback added, callbacks list: {conn.video.track_callbacks}")
                except Exception as e:
                    logging.error(f"Error in WebRTC connection: {e}")
                    print(f"[DEBUG] Exception in setup: {e}")
                    import traceback
                    traceback.print_exc()

            # Run the setup coroutine and then start the event loop
            loop.run_until_complete(setup())
            loop.run_forever()

        # Create a new event loop for the asyncio code
        loop = asyncio.new_event_loop()

        # Start the asyncio event loop in a separate thread
        asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,))
        asyncio_thread.start()

        try:
            display_count = 0
            while True:
                if not frame_queue.empty():
                # if frame_queue:
                    display_image = frame_queue.get()
                    # frame = frame_queue.pop()
                    # Convert BGR to RGB for display
                    display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
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
                                key="webrtc",
                                interpolate=True,
                            ),
                        ],
                        to="bgChildren",
                    )
                    display_count += 1
                    if display_count % 30 == 0:
                        print(f"[DEBUG] Displayed {display_count} frames, shape: {display_image.shape}")
                else:
                    # Sleep briefly to prevent high CPU usage
                    await asyncio.sleep(0.016)

                await asyncio.sleep(0.016)

        finally:
            # cv2.destroyAllWindows()
            # Stop the asyncio event loop
            loop.call_soon_threadsafe(loop.stop)
            asyncio_thread.join()


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