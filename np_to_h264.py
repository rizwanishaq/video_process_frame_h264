from typing import Optional
import cv2
import numpy as np
import PyNvCodec as nvc


class ColorConverter:

    def __init__(self, width, height, gpuid):
        # vpf does not support rgb_planar -> nv12, so we need the following chain:
        #     rgb_planar -> rgb -> yuv420 -> nv12
        self.to_rgb = nvc.PySurfaceConverter(
            width, height, nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.RGB, gpuid)
        self.to_yuv420 = nvc.PySurfaceConverter(
            width, height, nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420, gpuid)
        self.to_nv12 = nvc.PySurfaceConverter(
            width, height, nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12, gpuid)
        self.context = nvc.ColorspaceConversionContext(
            nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)
        self.uploader = nvc.PyFrameUploader(
            width, height, nvc.PixelFormat.RGB_PLANAR, gpuid)

    def convert(self, image):
        # convert opencv-style image to rgb_planar
        image = image[:, :, ::-1].transpose((2, 0, 1))
        image = np.ascontiguousarray(image)
        # copy data: cpu -> gpu
        surface = self.uploader.UploadSingleFrame(image)
        if surface.Empty():
            return None
        # do actual convertions
        surface = self.to_rgb.Execute(surface, self.context)
        if surface.Empty():
            return None
        surface = self.to_yuv420.Execute(surface, self.context)
        if surface.Empty():
            return None
        surface = self.to_nv12.Execute(surface, self.context)
        if surface.Empty():
            return None
        return surface


class VideoEncoder:
    def __init__(self, input_file: str, output_file: str, gpu_id: int = 0) -> None:
        """
        Initialize the VideoEncoder.

        Parameters:
            input_file (str): Path to the input video file.
            output_file (str): Path to the output encoded video file.
            gpu_id (int, optional): GPU ID to use for encoding. Defaults to 0.
        """
        self.video_stream = cv2.VideoCapture(input_file)

        # Get the video properties
        _, frame = self.video_stream.read()
        self.frame_height, self.frame_width, _ = frame.shape
        self.frame_fps = 25

        # Initialize the encoder
        self.encoder_params = {
            "preset": "P5",
            "tuning_info": "high_quality",
            "profile": "high",
            "codec": "h264",
            "s": f"{self.frame_width}x{self.frame_height}",
            "fps": str(self.frame_fps),
            "gop": str(self.frame_fps),
        }
        self.encoder = nvc.PyNvEncoder(self.encoder_params, gpu_id=gpu_id)

        self.output_file = output_file
        self.converter = ColorConverter(self.frame_width, self.frame_height, 0)

    def _cvt_color_BGR2YUV_NV12(self, image: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to YUV NV12 format.

        Parameters:
            image (np.ndarray): Input BGR image.

        Returns:
            np.ndarray: YUV NV12 formatted image.
        """
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
        uuvv = yuv[self.frame_height:].reshape(2, -1)
        uvuv = np.transpose(uuvv, axes=(1, 0))
        yuv[self.frame_height:] = uvuv.reshape(-1, self.frame_width)
        return yuv

    def _encode_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Encode a single frame and return the encoded packet.

        Parameters:
            frame (np.ndarray): Input frame.

        Returns:
            Optional[np.ndarray]: Encoded packet if available, otherwise None.
        """
        # nv12 = self._cvt_color_BGR2YUV_NV12(frame)

        nv12 = self.converter.convert(frame)
        packet = np.ndarray(shape=(0), dtype=np.uint8)

        if self.encoder.EncodeSingleFrame(nv12, packet):
            print(packet)
            # return packet
        return None

    def _write_packet(self, file, packet: np.ndarray) -> None:
        """
        Write the packet to the output file.

        Parameters:
            file: File object.
            packet (np.ndarray): Encoded packet.
        """
        file.write(packet.data)
        # file.write(bytearray(packet))

    def encode_video(self) -> None:
        """
        Encode the input video and save the result to the output file.
        """
        with open(self.output_file, "wb") as dstfile:
            while True:
                ret, frame = self.video_stream.read()
                if not ret:
                    break
                packet = self._encode_frame(frame)
                if packet is not None:
                    self._write_packet(dstfile, packet)

            packet = np.ndarray(shape=(0), dtype=np.uint8)
            while self.encoder.FlushSinglePacket(packet):
                self._write_packet(dstfile, packet)


if __name__ == "__main__":
    input_file_path = "abc.mp4"
    output_file_path = "output.h264"
    gpu_id = 0

    video_encoder = VideoEncoder(input_file_path, output_file_path, gpu_id)
    video_encoder.encode_video()
