import torch
import torchvision
import numpy as np
import PIL
import random
import cv2
import os
from typing import Optional, Iterable


__all__ = ['VideoFilePathToTensor',
           'VideoFolderPathToTensor',
           'VideoResize',
           'VideoRandomCrop',
           'VideoCenterCrop',
           'VideoRandomHorizontalFlip',
           'VideoRandomVerticalFlip',
           'VideoGrayscale']


class VideoFilePathToTensor:
    """ load video at given file path to torch.Tensor (C x L x H x W, C = 3) 
        It can be composed with torchvision.transforms.Compose().
        
    Args:
        max_len (int): Maximum output frames count (L <= max_len). Default is None.
            If it is set to None, it will output all frames. 
        fps (int): sample frame per seconds. It must be lower than or equal to the original video fps.
            Default is None. 
        padding_mode (str): Type of padding. Default to None. Only available when max_len is not None.
            - None: won't padding, video length is variable.
            - 'zero': padding the rest empty frames to zeros.
            - 'last': padding the rest empty frames to the last frame.
    """

    def __init__(self, max_len: Optional[int] = None, fps: Optional[int] = None, padding_mode: Optional[str] = None):
        self.max_len = max_len
        self.fps = fps
        assert padding_mode in (None, 'zero', 'last')
        self.padding_mode = padding_mode
        self.channels = 3  # only available to read 3 channels video

    def __call__(self, path: str) -> torch.Tensor:
        """
        Args:
            path (str): path of video file.
            
        Returns:
            torch.Tensor: Video Tensor (C x L x H x W)
        """

        # Open video file
        cap = cv2.VideoCapture(path)
        assert cap.isOpened()

        # Calculate sample_factor to reset fps
        sample_factor = 1
        if self.fps:
            old_fps = cap.get(cv2.CAP_PROP_FPS)  # fps of video
            sample_factor = int(old_fps / self.fps)
            assert sample_factor >= 1
        
        # Init empty output frames (C x L x H x W)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate desired frames count
        time_len = None
        if self.max_len:
            # Time length has upper bound
            if self.padding_mode:
                # Padding all video to the same time length
                time_len = self.max_len
            else:
                # Video have variable time length
                time_len = min(int(num_frames / sample_factor), self.max_len)
        else:
            # Time length is unlimited
            time_len = int(num_frames / sample_factor)
        assert time_len is not None
        print(f'Total frames: {num_frames}, sample factor: {sample_factor}, {time_len} frames will be extracted.')

        frames = torch.FloatTensor(self.channels, time_len, height, width)

        for index in range(time_len):
            frame_index = sample_factor * index

            # read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                # Successfully read frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1)  # (H x W x C) to (C x H x W)
                frames[:, index, :, :] = frame.float()
            else:
                # Reach the end of the video
                if self.padding_mode == 'zero':
                    # fill the rest frames with 0.0
                    frames[:, index:, :, :] = 0
                elif self.padding_mode == 'last':
                    # fill the rest frames with the last frame
                    assert index > 0
                    frames[:, index:, :, :] = frames[:, index-1, :, :].view(self.channels, 1, height, width)
                break

        frames /= 255
        cap.release()
        return frames


class VideoFolderPathToTensor:
    """ load video at given folder path to torch.Tensor (C x L x H x W) 
        It can be composed with torchvision.transforms.Compose().
        
    Args:
        max_len (int): Maximum output time depth (L <= max_len). Default is None.
            If it is set to None, it will output all frames. 
        padding_mode (str): Type of padding. Default to None. Only available when max_len is not None.
            - None: won't padding, video length is variable.
            - 'zero': padding the rest empty frames to zeros.
            - 'last': padding the rest empty frames to the last frame.
    """

    def __init__(self, max_len: Optional[int] = None, padding_mode: Optional[str] = None):
        self.max_len = max_len
        assert padding_mode in (None, 'zero', 'last')
        self.padding_mode = padding_mode

    def __call__(self, path: str) -> torch.Tensor:
        """
        Args:
            path (str): path of video folder.
            
        Returns:
            torch.Tensor: Video Tensor (C x L x H x W)
        """
        
        # get video properity
        frames_path = sorted([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        frame = cv2.imread(frames_path[0])
        height, width, channels = frame.shape
        num_frames = len(frames_path)
        
        # init empty output frames (C x L x H x W)
        time_len = None
        if self.max_len:
            # time length has upper bound
            if self.padding_mode:
                # padding all video to the same time length
                time_len = self.max_len
            else:
                # video have variable time length
                time_len = min(num_frames, self.max_len)
        else:
            # time length is unlimited
            time_len = num_frames

        frames = torch.FloatTensor(channels, time_len, height, width)

        # load the video to tensor
        for index in range(time_len):
            if index < num_frames:
                # frame exists
                # read frame
                frame = cv2.imread(frames_path[index])
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1)
                frames[:, index, :, :] = frame.float()
            else:
                # reach the end of the video
                if self.padding_mode == 'zero':
                    # fill the rest frames with 0.0
                    frames[:, index:, :, :] = 0
                elif self.padding_mode == 'last':
                    # fill the rest frames with the last frame
                    assert index > 0
                    frames[:, index:, :, :] = frames[:, index-1, :, :].view(channels, 1, height, width)
                break

        frames /= 255
        return frames


class VideoResize:
    """ resize video tensor (C x L x H x W) to (C x L x h x w) 
    
    Args:
        size (sequence): Desired output size. size is a sequence like (H, W),
            output size will matched to this.
        interpolation (int, optional): Desired interpolation. Default is 'PIL.Image.BILINEAR'
    """

    def __init__(self, size, interpolation=PIL.Image.BILINEAR):
        assert isinstance(size, Iterable) and len(size) == 2
        self.size = size
        self.interpolation = interpolation

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video to be scaled (C x L x H x W)
        
        Returns:
            torch.Tensor: Rescaled video (C x L x h x w)
        """

        h, w = self.size
        C, L, H, W = video.size()
        rescaled_video = torch.FloatTensor(C, L, h, w)
        
        # use torchvision implementation to resize video frames
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(self.size, self.interpolation),
            torchvision.transforms.ToTensor(),
        ])

        for l in range(L):
            frame = video[:, l, :, :]
            frame = transform(frame)
            rescaled_video[:, l, :, :] = frame

        return rescaled_video

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

        
class VideoRandomCrop:
    """ Crop the given Video Tensor (C x L x H x W) at a random location.

    Args:
        size (sequence): Desired output size like (h, w).
    """

    def __init__(self, size):
        assert len(size) == 2
        self.size = size
    
    def __call__(self, video):
        """ 
        Args:
            video (torch.Tensor): Video (C x L x H x W) to be cropped.

        Returns:
            torch.Tensor: Cropped video (C x L x h x w).
        """

        H, W = video.size()[2:]
        h, w = self.size
        assert H >= h and W >= w

        top = np.random.randint(0, H - h)
        left = np.random.randint(0, W - w)

        video = video[:, :, top : top + h, left : left + w]
        
        return video


class VideoCenterCrop:
    """ Crops the given video tensor (C x L x H x W) at the center.

    Args:
        size (sequence): Desired output size of the crop like (h, w).
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video (C x L x H x W) to be cropped.
        
        Returns:
            torch.Tensor: Cropped Video (C x L x h x w).
        """

        H, W = video.size()[2:]
        h, w = self.size
        assert H >= h and W >= w
        
        top = int((H - h) / 2)
        left = int((W - w) / 2)

        video = video[:, :, top : top + h, left : left + w]
        
        return video


class VideoRandomHorizontalFlip:
    """ Horizontal flip the given video tensor (C x L x H x W) randomly with a given probability.

    Args:
        p (float): probability of the video being flipped. Default value is 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video to flipped.
        
        Returns:
            torch.Tensor: Randomly flipped video.
        """

        if random.random() < self.p:
            # horizontal flip the video
            video = video.flip([3])

        return video
            

class VideoRandomVerticalFlip:
    """ Vertical flip the given video tensor (C x L x H x W) randomly with a given probability.

    Args:
        p (float): probability of the video being flipped. Default value is 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video to flipped.
        
        Returns:
            torch.Tensor: Randomly flipped video.
        """

        if random.random() < self.p:
            # horizontal flip the video
            video = video.flip([2])

        return video


class VideoGrayscale:
    """ Convert video (C x L x H x W) to grayscale (C' x L x H x W, C' = 1 or 3)

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output video
    """

    def __init__(self, num_output_channels=1):
        assert num_output_channels in (1, 3)
        self.num_output_channels = num_output_channels

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video (3 x L x H x W) to be converted to grayscale.

        Returns:
            torch.Tensor: Grayscaled video (1 x L x H x W  or  3 x L x H x W)
        """

        C, L, H, W = video.size()
        grayscaled_video = torch.FloatTensor(self.num_output_channels, L, H, W)
        
        # use torchvision implemention to convert video frames to gray scale
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Grayscale(self.num_output_channels),
            torchvision.transforms.ToTensor(),
        ])

        for l in range(L):
            frame = video[:, l, :, :]
            frame = transform(frame)
            grayscaled_video[:, l, :, :] = frame

        return grayscaled_video


        