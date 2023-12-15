#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import PyNvCodec as nvc


class ColorConverter:
    """
    ColorConverter class for converting images between different pixel formats.

    Args:
        width (int): Width of the image.
        height (int): Height of the image.
        gpuid (int): GPU ID to use for conversion.

    Attributes:
        to_rgb (PySurfaceConverter): Converter from RGB_PLANAR to RGB.
        to_yuv420 (PySurfaceConverter): Converter from RGB to YUV420.
        to_nv12 (PySurfaceConverter): Converter from YUV420 to NV12.
        context (ColorspaceConversionContext): Colorspace conversion context.
        uploader (PyFrameUploader): Frame uploader from CPU to GPU.
    """

    def __init__(self, width, height, gpuid):
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

    def convert(self, image: np.ndarray):
        """
        Convert the given image to NV12 format.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            nvc.PySurface: Converted image in NV12 format.
        """
        image = image[:, :, ::-1].transpose((2, 0, 1))
        image = np.ascontiguousarray(image)
        surface = self.uploader.UploadSingleFrame(image)
        if surface.Empty():
            return None

        operations = [self.to_rgb, self.to_yuv420, self.to_nv12]
        for operation in operations:
            surface = operation.Execute(surface, self.context)
            if surface.Empty():
                return None

        return surface


def main():
    video_stream = cv2.VideoCapture("abc.mp4")
    ret, frame = video_stream.read()
    frame_height, frame_width, _ = frame.shape
    gpuID = 0
    frame_fps = 30

    encoder = nvc.PyNvEncoder({
        "preset": "P5",
        "tuning_info": "high_quality",
        "profile": "high",
        "codec": "h264",
        "s": f"{frame_width}x{frame_height}",
        "fps": str(frame_fps),
        "gop": str(frame_fps),
    }, gpu_id=0)

    converter = ColorConverter(frame_width, frame_height, 0)
    with open("output.h264", "wb") as dstfile:
        while True:
            nv12 = converter.convert(frame)
            packet = np.ndarray(shape=(0), dtype=np.uint8)
            if encoder.EncodeSingleSurface(nv12, packet):
                dstfile.write(packet.data)

            ret, frame = video_stream.read()
            if not ret:
                break

        packet = np.ndarray(shape=(0), dtype=np.uint8)
        while encoder.FlushSinglePacket(packet):
            dstfile.write(packet.data)


if __name__ == "__main__":
    main()
