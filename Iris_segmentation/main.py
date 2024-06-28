import cv2
import numpy as np
import os

def blur_white_mask(mask):
  """
  Hàm làm mờ phần màu trắng trong ảnh mask.
  """
  # Chuyển đổi ảnh mask sang hệ màu HSV
  hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
  # Tách kênh V (Giá trị)
  value = hsv[:, :, 2]
  # Áp dụng bộ lọc Gaussian để làm mờ kênh V
  blurred_value = cv2.GaussianBlur(value, (5, 5), 0)
  # Gán kênh V đã làm mờ vào ảnh HSV
  hsv[:, :, 2] = blurred_value
  # Chuyển đổi ảnh HSV về hệ màu BGR
  blurred_mask = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
  return blurred_mask

def overlay_images(img, mask):
  """
  Hàm lồng ghép ảnh gốc và ảnh mask.
  """
  # Chuyển đổi ảnh mask sang grayscale
  gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
  # Tìm các pixel khác màu đen trong ảnh mask
  white_pixels = np.where(gray_mask == 0)
  # Lấy các pixel tương ứng từ ảnh gốc
  overlaid_img = img.copy()
  overlaid_img[white_pixels[0], white_pixels[1]] = mask[white_pixels[0], white_pixels[1]]
  return overlaid_img

def process_image(image):
    # Tìm giá trị pixel tối đa
    max_value = np.max(image)
    # Tìm các pixel khác màu đen
    non_black_pixels = np.where(image != 0)
    # Tìm giới hạn trên, dưới, trái, phải của các pixel khác màu đen
    top = np.min(non_black_pixels[0])
    bottom = np.max(non_black_pixels[0])
    left = np.min(non_black_pixels[1])
    right = np.max(non_black_pixels[1])
    # Tính chiều dài và chiều rộng của hình chữ nhật
    width = right - left
    height = bottom - top
    # Nếu chiều rộng lớn hơn chiều cao, mở rộng hình chữ nhật theo chiều dọc
    if width > height:
        diff = width - height
        top -= diff // 2
        bottom += diff // 2
    # Ngược lại, mở rộng hình chữ nhật theo chiều ngang
    else:
        diff = height - width
        left -= diff // 2
        right += diff // 2
    # Cắt hình vuông từ ảnh gốc
    square_img = image[top:bottom, left:right]
    # Kiểm tra xem square_img có tồn tại và không rỗng
    if square_img is not None and square_img.size != 0:
        # Thay đổi kích thước hình vuông thành 200x200
        resized_img = cv2.resize(square_img, (200, 200))
    else:
        print("square_img is empty or does not exist")
        return None  # Trả về None nếu square_img là rỗng hoặc không tồn tại

    # Trả về ảnh sau khi xử lý
    return resized_img

