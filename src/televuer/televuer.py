from vuer import Vuer
from vuer.schemas import ImageBackground, Hands, MotionControllers, WebRTCVideoPlane, WebRTCStereoVideoPlane
from multiprocessing import Value, Array, Process, shared_memory
import numpy as np
import asyncio
import cv2
import os
from pathlib import Path


class TeleVuer:
<<<<<<< HEAD
    def __init__(self, use_hand_tracking: bool, pass_through:bool=False, binocular: bool=True, img_shape: tuple=None, 
                       cert_file=None, key_file=None, webrtc: bool=False, webrtc_url: str=None):
=======
    def __init__(self, binocular: bool, use_hand_tracking: bool, img_shape, img_shm_name, cert_file=None, key_file=None, ngrok=False, webrtc=False):
>>>>>>> de8286aa4b1a3ec37002484a5957b0cb3baf8ab1
        """
        TeleVuer class for OpenXR-based XR teleoperate applications.
        This class handles the communication with the Vuer server and manages image and pose data.

        :param use_hand_tracking: bool, whether to use hand tracking or controller tracking.
<<<<<<< HEAD
        :param pass_through: bool, controls the VR viewing mode.

            Note:
            - if pass_through is True, the XR user will see the real world through the VR headset cameras.
            - if pass_through is False, the XR user will see the images provided by webrtc or render_to_xr method:
            - webrtc is prior to render_to_xr. if webrtc is True, the class will use webrtc for image transmission.
            - if webrtc is False, the class will use render_to_xr for image transmission.
    
        :param binocular: bool, whether the application is binocular (stereoscopic) or monocular.
        :param img_shape: tuple, shape of the head image (height, width).
        :param cert_file: str, path to the SSL certificate file.
        :param key_file: str, path to the SSL key file.
        :param webrtc: bool, whether to use WebRTC for real-time communication. if False, use ImageBackground.
        :param webrtc_url: str, URL for the WebRTC offer.
=======
        :param img_shape: tuple, shape of the image (height, width, channels).
        :param img_shm_name: str, name of the shared memory for the image.
        :param cert_file: str, path to the SSL certificate file.
        :param key_file: str, path to the SSL key file.
        :param ngrok: bool, whether to use ngrok for tunneling.
>>>>>>> de8286aa4b1a3ec37002484a5957b0cb3baf8ab1
        """
        self.use_hand_tracking = use_hand_tracking
<<<<<<< HEAD
        self.pass_through = pass_through

        self.binocular = binocular
        self.img_shape = (img_shape[0], img_shape[1], 3)
        self.img_height = self.img_shape[0]
=======
        self.img_height = img_shape[0]
>>>>>>> de8286aa4b1a3ec37002484a5957b0cb3baf8ab1
        if self.binocular:
            self.img_width  = img_shape[1] // 2
        else:
<<<<<<< HEAD
            self.img_width  = self.img_shape[1]
        self.aspect_ratio = self.img_width / self.img_height

=======
            self.img_width  = img_shape[1]
        
>>>>>>> de8286aa4b1a3ec37002484a5957b0cb3baf8ab1
        current_module_dir = Path(__file__).resolve().parent.parent.parent
        if cert_file is None:
            cert_file = os.path.join(current_module_dir, "cert.pem")
        if key_file is None:
            key_file = os.path.join(current_module_dir, "key.pem")

        self.vuer = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)
        self.vuer.add_handler("CAMERA_MOVE")(self.on_cam_move)
        if self.use_hand_tracking:
            self.vuer.add_handler("HAND_MOVE")(self.on_hand_move)
        else:
            self.vuer.add_handler("CONTROLLER_MOVE")(self.on_controller_move)

<<<<<<< HEAD
        self.webrtc = webrtc
        self.webrtc_url = webrtc_url

        if self.webrtc:
            if self.binocular:
                self.vuer.spawn(start=False)(self.main_image_binocular_webrtc)
            else:
                self.vuer.spawn(start=False)(self.main_image_monocular_webrtc)
        else:
            if self.pass_through is False:
                self.img2display_shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
                self.img2display = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.img2display_shm.buf)
                self.latest_frame = None
                self.new_frame_event = threading.Event()
                self.stop_writer_event = threading.Event()
                self.writer_thread = threading.Thread(target=self._xr_render_loop, daemon=True)
                self.writer_thread.start()
            if self.binocular:
                 self.vuer.spawn(start=False)(self.main_image_binocular)
            else:
                 self.vuer.spawn(start=False)(self.main_image_monocular)
=======
        existing_shm = shared_memory.SharedMemory(name=img_shm_name)
        self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=existing_shm.buf)

        self.webrtc = webrtc
        if self.binocular and not self.webrtc:
            self.vuer.spawn(start=False)(self.main_image_binocular)
        elif not self.binocular and not self.webrtc:
            self.vuer.spawn(start=False)(self.main_image_monocular)
        elif self.webrtc:
            self.vuer.spawn(start=False)(self.main_image_webrtc)
>>>>>>> de8286aa4b1a3ec37002484a5957b0cb3baf8ab1

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
<<<<<<< HEAD
    
    def _vuer_run(self):
        try:
            self.vuer.run()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Vuer encountered an error: {e}")
        finally:
            if hasattr(self, "stop_writer_event"):
                self.stop_writer_event.set()

    def _xr_render_loop(self):
        while not self.stop_writer_event.is_set():
            if not self.new_frame_event.wait(timeout=0.1):
                continue
            self.new_frame_event.clear()
            if self.latest_frame is None:
                continue
            latest_frame = self.latest_frame
            latest_frame = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
            self.img2display[:] = latest_frame
    
    def render_to_xr(self, image):
        if self.webrtc or self.pass_through:
            print("[TeleVuer] Warning: render_to_xr is ignored when webrtc is enabled or pass_through is True.")
            return
        self.latest_frame = image
        self.new_frame_event.set()

    def close(self):
        self.process.terminate()
        self.process.join(timeout=0.5)
        if not self.webrtc and not self.pass_through:
            self.stop_writer_event.set()
            self.new_frame_event.set()
            self.writer_thread.join(timeout=0.5)
            try:
                self.img2display_shm.close()
                self.img2display_shm.unlink()
            except:
                pass
=======

    def vuer_run(self):
        self.vuer.run()
>>>>>>> de8286aa4b1a3ec37002484a5957b0cb3baf8ab1

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
    
    async def main_image_binocular(self, session):
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
<<<<<<< HEAD
            if self.pass_through is False:
                session.upsert(
                    [
                        ImageBackground(
                            self.img2display[:, :self.img_width],
                            aspect=self.aspect_ratio,
                            height=1,
                            distanceToCamera=1,
                            # The underlying rendering engine supported a layer binary bitmask for both objects and the camera. 
                            # Below we set the two image planes, left and right, to layers=1 and layers=2. 
                            # Note that these two masks are associated with left eye’s camera and the right eye’s camera.
                            layers=1,
                            format="jpeg",
                            quality=80,
                            key="background-left",
                            interpolate=True,
                        ),
                        ImageBackground(
                            self.img2display[:, self.img_width:],
                            aspect=self.aspect_ratio,
                            height=1,
                            distanceToCamera=1,
                            layers=2,
                            format="jpeg",
                            quality=80,
                            key="background-right",
                            interpolate=True,
                        ),
                    ],
                    to="bgChildren",
                )
=======
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
>>>>>>> de8286aa4b1a3ec37002484a5957b0cb3baf8ab1
            # 'jpeg' encoding should give you about 30fps with a 16ms wait in-between.
            await asyncio.sleep(0.016 * 2)

    async def main_image_monocular(self, session):
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
<<<<<<< HEAD
            if self.pass_through is False:
                session.upsert(
                    [
                        ImageBackground(
                            self.img2display,
                            aspect=self.aspect_ratio,
                            height=1,
                            distanceToCamera=1,
                            format="jpeg",
                            quality=80,
                            key="background-mono",
                            interpolate=True,
                        ),
                    ],
                    to="bgChildren",
                )
            await asyncio.sleep(0.016)

    async def main_image_binocular_webrtc(self, session):
=======
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
>>>>>>> de8286aa4b1a3ec37002484a5957b0cb3baf8ab1
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
<<<<<<< HEAD
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            if self.pass_through is False:
                session.upsert(
                    WebRTCStereoVideoPlane(
                        src=self.webrtc_url,
                        iceServer=None,
                        iceServers=[], 
                        key="video-quad",
                        aspect=self.aspect_ratio,
                        height = 7,
                        layout="stereo-left-right"
                    ),
                    to="bgChildren",
                )

            await asyncio.sleep(0.016)

    async def main_image_monocular_webrtc(self, session):
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
            if self.pass_through is False:
                session.upsert(
                    WebRTCVideoPlane(
                        src=self.webrtc_url,
                        iceServer=None,
                        iceServers=[],
                        key="video-quad",
                        aspect=self.aspect_ratio,
                        height = 7,
                    ),
                    to="bgChildren",
                )

            await asyncio.sleep(0.016)
=======
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

        frame_queue = Queue()
        # frame_queue = deque(maxlen=1) 

        # Choose a connection method (uncomment the correct one)
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.123.161")
        # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="10.1.10.48")

        # Async function to receive video frames and put them in the queue
        async def recv_camera_stream(track: MediaStreamTrack):
            while True:
                print("trying to get frame")
                frame = await track.recv()
                print("recieved frame")
                # Convert the frame to a NumPy array
                img = frame.to_ndarray(format="rgb24")
                frame_queue.put(frame)
                # frame_queue.append(frame)

        def run_asyncio_loop(loop):
            print("start run_asyncio_loop")
            asyncio.set_event_loop(loop)
            print("setting run_asyncio_loop")
            async def setup():
                try:
                    print("start setup")
                    # Connect to the device
                    await conn.connect()
                    print("done connecting")

                    # Switch video channel on and start receiving video frames
                    conn.video.switchVideoChannel(True)
                    print("successfully setting video channel")

                    # Add callback to handle received video frames
                    conn.video.add_track_callback(recv_camera_stream)
                except Exception as e:
                    logging.error(f"Error in WebRTC connection: {e}")

            # Run the setup coroutine and then start the event loop
            loop.run_until_complete(setup())
            print("coroutine done")
            loop.run_forever()

        # Create a new event loop for the asyncio code
        loop = asyncio.new_event_loop()

        # Start the asyncio event loop in a separate thread
        asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,))
        asyncio_thread.start()

        try:

            while True:
                if not frame_queue.empty():
                # if frame_queue:
                    frame = frame_queue.get()
                    # frame = frame_queue.pop()
                    # display_image = frame.to_ndarray(format="rgb24")

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
                else:
                    # Sleep briefly to prevent high CPU usage
                    await asyncio.sleep(0.016)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                await asyncio.sleep(0.016)

        finally:
            # cv2.destroyAllWindows()
            # Stop the asyncio event loop
            loop.call_soon_threadsafe(loop.stop)
            asyncio_thread.join()


>>>>>>> de8286aa4b1a3ec37002484a5957b0cb3baf8ab1
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