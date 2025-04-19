import cv2
import math
import numpy as np

def insert_image(background, overlay, x, y, angle, scale, clip=True):
    """
    Inserts an image onto a background image with rotation and scaling, ensuring no clipping
    and transparent background for rotated padding.

    Args:
        background (numpy.ndarray): Background image (BGR or BGRA).
        overlay (numpy.ndarray): Overlay image (BGR or BGRA).
        x (int): x-coordinate of the top-left corner of the overlay in the background.
        y (int): y-coordinate of the top-left corner of the overlay in the background.
        angle (float): Rotation angle in degrees.
        scale (float): Scaling factor.

    Returns:
        numpy.ndarray: Combined image with overlay inserted.
    """
    background = background.copy()

    # Scale the overlay image
    new_width = int(overlay.shape[1] * scale)
    new_height = int(overlay.shape[0] * scale)
    overlay_resized = cv2.resize(overlay, (new_width, new_height))

    # Rotate the overlay image with padding and alpha channel
    overlay_rotated, bounding_box = rotate_image(overlay_resized, angle)

    if False:
        # Handle transparency for the overlay_rotated
        if overlay_rotated.shape[2] == 4:
            overlay_rgb = overlay_rotated[:, :, :3]
            alpha = overlay_rotated[:, :, 3] / 255.0
        else:
            overlay_rgb = overlay_rotated
            alpha = np.ones(overlay_rotated.shape[:2], dtype=np.float32)

        # Region of interest for the overlay in the background
        y1 = y
        y2 = y + overlay_rotated.shape[0]
        x1 = x
        x2 = x + overlay_rotated.shape[1]
        # Adjust bounding box so it still represents the overlay_resized position in the background
        bounding_box = (
            bounding_box[0] + x,
            bounding_box[1] + y,
            bounding_box[2] + x,
            bounding_box[3] + y,
            bounding_box[4] + x,
            bounding_box[5] + y,
            bounding_box[6] + x,
            bounding_box[7] + y
        )
        if y1 < 0 or y2 > background.shape[0] or x1 < 0 or x2 > background.shape[1]:
            return background, bounding_box
        
        # Ensure background has an alpha channel
        if background.shape[2] == 3:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

        # Extract the background region
        bg_region = background[y1:y2, x1:x2]
        bg_alpha = bg_region[:, :, 3] / 255.0 if background.shape[2] == 4 else np.ones(bg_region.shape[:2], dtype=np.float32)


        # Composite the overlay onto the background
        for c in range(3):
            bg_region[:, :, c] = (overlay_rgb[:, :, c] * alpha + bg_region[:, :, c] * (1 - alpha) * bg_alpha) / (alpha + (1 - alpha) * bg_alpha)
        bg_region[:, :, 3] = (alpha * 255 + (1 - alpha) * bg_region[:, :, 3]).astype(np.uint8)
        background[y1:y2, x1:x2] = bg_region
        
        return background, bounding_box
    else:
        # Calculate new background dimensions to avoid clipping
        overlay_h, overlay_w = overlay_rotated.shape[:2]
        x_min = min(x, 0)
        x_max = max(x + overlay_w, background.shape[1])
        y_min = min(y, 0)
        y_max = max(y + overlay_h, background.shape[0])

        new_width_bg = x_max - x_min
        new_height_bg = y_max - y_min

        # Create new background with alpha channel initialized to transparent
        new_bg = np.zeros((new_height_bg, new_width_bg, 4), dtype=np.uint8)
        new_bg[:, :, 3] = 0  # Transparent by default

        # Calculate offset to place original background into new_bg
        offset_x = -x_min
        offset_y = -y_min

        # Determine the region of the original background to place into new_bg
        bg_src_x_start = max(0, -offset_x)
        bg_src_y_start = max(0, -offset_y)
        bg_src_x_end = bg_src_x_start + min(background.shape[1], new_width_bg - offset_x)
        bg_src_y_end = bg_src_y_start + min(background.shape[0], new_height_bg - offset_y)

        bg_dst_x_start = max(0, offset_x)
        bg_dst_y_start = max(0, offset_y)
        bg_dst_x_end = bg_dst_x_start + (bg_src_x_end - bg_src_x_start)
        bg_dst_y_end = bg_dst_y_start + (bg_src_y_end - bg_src_y_start)

        # Place the original background into the new_bg
        if background.shape[2] == 4:
            new_bg[bg_dst_y_start:bg_dst_y_end, bg_dst_x_start:bg_dst_x_end] = background[bg_src_y_start:bg_src_y_end, bg_src_x_start:bg_src_x_end]
        else:
            new_bg[bg_dst_y_start:bg_dst_y_end, bg_dst_x_start:bg_dst_x_end, :3] = background[bg_src_y_start:bg_src_y_end, bg_src_x_start:bg_src_x_end]
            new_bg[bg_dst_y_start:bg_dst_y_end, bg_dst_x_start:bg_dst_x_end, 3] = 255  # Opaque

        # Adjust x and y for the new background
        new_x = x - x_min
        new_y = y - y_min

        # Handle transparency for the overlay_rotated
        if overlay_rotated.shape[2] == 4:
            overlay_rgb = overlay_rotated[:, :, :3]
            alpha = overlay_rotated[:, :, 3] / 255.0
        else:
            overlay_rgb = overlay_rotated
            alpha = np.ones(overlay_rotated.shape[:2], dtype=np.float32)

        # Region of interest for the overlay in the new_bg
        y1 = new_y
        y2 = new_y + overlay_rotated.shape[0]
        x1 = new_x
        x2 = new_x + overlay_rotated.shape[1]
        bounding_box = (
            bounding_box[0] + new_x,
            bounding_box[1] + new_y,
            bounding_box[2] + new_x,
            bounding_box[3] + new_y,
            bounding_box[4] + new_x,
            bounding_box[5] + new_y,
            bounding_box[6] + new_x,
            bounding_box[7] + new_y
        )
        if y1 < 0 or y2 > new_bg.shape[0] or x1 < 0 or x2 > new_bg.shape[1]:
            return new_bg, bounding_box  # This should not happen due to earlier expansion

        # Extract the background region
        bg_region = new_bg[y1:y2, x1:x2]
        bg_alpha = bg_region[:, :, 3] / 255.0

        # Composite the overlay onto the background
        for c in range(3):
            bg_region[:, :, c] = (overlay_rgb[:, :, c] * alpha + bg_region[:, :, c] * (1 - alpha) * bg_alpha) / (alpha + (1 - alpha) * bg_alpha)
        bg_region[:, :, 3] = (alpha * 255 + (1 - alpha) * bg_region[:, :, 3]).astype(np.uint8)

        new_bg[y1:y2, x1:x2] = bg_region

        if clip:
            # Clip the new background to the original size
            new_bg = new_bg[bg_dst_y_start:bg_dst_y_end, bg_dst_x_start:bg_dst_x_end]
            new_bg = cv2.resize(new_bg, (background.shape[1], background.shape[0]), interpolation=cv2.INTER_LINEAR)

        return new_bg, bounding_box

def rotate_image(image, angle_degrees, center=None):
    """
    Rotates an image by a given angle with padding and transparent background.

    Args:
        image (numpy.ndarray): Input image (BGR or BGRA).
        angle_degrees (float): Rotation angle in degrees.
        center (tuple): Optional center of rotation.

    Returns:
        numpy.ndarray: Rotated image with transparent padding (BGRA).
    """
    # Convert angle to radians
    angle_radians = math.radians(angle_degrees)
    original_height, original_width = image.shape[:2]

    # Add alpha channel if missing
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    if center is None:
        center = (original_width // 2, original_height // 2)

    # Calculate new image dimensions after rotation
    cos_theta = abs(math.cos(angle_radians))
    sin_theta = abs(math.sin(angle_radians))
    new_width = int((original_height * sin_theta) + (original_width * cos_theta))
    new_height = int((original_height * cos_theta) + (original_width * sin_theta))

    # Adjust rotation matrix to account for translation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # Perform rotation with transparent padding
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    # Calculate bounding box corners of the rotated image (x1, y1, x2, y2, x3, y3, x4, y4)
    corners = np.array([[0, 0], [original_width, 0], [original_width, original_height], [0, original_height]], dtype=np.float32)
    corners = cv2.transform(np.array([corners]), rotation_matrix)[0]
    x1, y1 = corners[0]
    x2, y2 = corners[1]
    x3, y3 = corners[2]
    x4, y4 = corners[3]
    bounding_box = (int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4))

    return rotated_image, bounding_box
