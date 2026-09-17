# -*- coding: utf-8 -*-
import SimpleITK as sitk
import os
import cv2
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import shutil

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brain_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 主处理函数
def mha_to_highres(save_dir, mha_path, wc=None, ws=None, upscale_factor=2,
                   output_format='png', jpeg_quality=95, modality='CT',
                   window_preset='auto', enable_clahe=True, correct_bias_field=False,
                   blank_threshold=0.01, remove_black_images=True):
    """
    增强版脑部影像处理函数（含全黑切片检测）
    新增参数：
        remove_black_images: 是否启用全黑图像清理
    """
    try:
        # 输入验证
        validate_inputs(mha_path, save_dir, upscale_factor, output_format)

        # 创建输出目录
        os.makedirs(save_dir, exist_ok=True, mode=0o755)
        logger.info(f"输出目录已创建: {save_dir}")

        # 读取图像数据
        image = sitk.ReadImage(mha_path)
        spacing = image.GetSpacing()

        # 偏置场校正（仅MR）
        if correct_bias_field and modality == 'MR':
            logger.info("执行N4偏置场校正...")
            image = n4_bias_correction(image)

        img_data = sitk.GetArrayFromImage(image)
        logger.info(f"成功读取图像，尺寸: {img_data.shape} 间距: {spacing}")

        # 获取窗宽窗位设置
        wc, ws = get_window_settings(img_data, modality, window_preset, wc, ws)
        low, high = wc - ws / 2, wc + ws / 2
        logger.info(f"使用窗位={wc}, 窗宽={ws} [显示范围({low:.1f}-{high:.1f})]")

        # 准备并行处理
        process_slice = partial(
            process_and_save_slice,
            save_dir=save_dir,
            low=low,
            high=high,
            upscale_factor=upscale_factor,
            modality=modality,
            output_format=output_format,
            jpeg_quality=jpeg_quality,
            enable_clahe=enable_clahe,
            original_spacing=spacing,
            blank_threshold=blank_threshold
        )

        # 并行处理切片
        with ThreadPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
            results = executor.map(process_slice, enumerate(img_data))

        # 统计结果
        saved_count = sum(results)
        logger.info(f"初步处理完成，保存 {saved_count}/{img_data.shape[0]} 个切片")

        # 全黑图像清理
        final_count = saved_count
        if remove_black_images:
            if output_format.lower() in ('jpg', 'jpeg'):
                removed = remove_black_jpg(save_dir, blank_threshold)
            else:
                removed = remove_blank_images(save_dir, blank_threshold)
            final_count = saved_count - removed
            logger.info(f"最终有效切片: {final_count} (移除{removed}个空白图像)")

        return final_count

    except Exception as e:
        logger.error(f"处理失败: {str(e)}", exc_info=True)
        raise


# 辅助函数
def validate_inputs(mha_path, save_dir, upscale_factor, output_format):
    """输入验证"""
    if not os.path.isfile(mha_path):
        raise FileNotFoundError(f"输入文件不存在: {mha_path}")
    if upscale_factor < 1 or not isinstance(upscale_factor, int):
        raise ValueError("上采样因子必须为正整数")
    if output_format.lower() not in ['png', 'jpg', 'jpeg']:
        raise ValueError("仅支持 PNG 和 JPEG 格式")
    parent_dir = os.path.dirname(save_dir) or "."
    os.makedirs(parent_dir, exist_ok=True)
    if not os.access(parent_dir, os.W_OK):
        raise PermissionError(f"无写入权限: {parent_dir}")


def get_window_settings(img_data, modality, window_preset, user_wc, user_ws):
    """获取脑部专用窗设置"""
    if user_wc is not None and user_ws is not None:
        return user_wc, user_ws

    PRESETS = {
        'CT': {
            'ct_brain': (40, 80),
            'ct_stroke': (35, 110),
            'ct_bone': (600, 2800),
            'ct_subdural': (80, 200)
        },
        'MR': {
            'mr_t1': (450, 900),
            'mr_t2': (3000, 6000),
            'mr_flair': (5000, 10000),
            'mr_swi': (40, 150)
        }
    }

    if window_preset == 'auto':
        if modality == 'CT':
            return auto_detect_ct_window(img_data)
        else:
            return auto_detect_mr_window(img_data)

    try:
        return PRESETS[modality][window_preset]
    except KeyError:
        raise ValueError(f"不支持的预设: {modality}_{window_preset}")


def auto_detect_ct_window(img_data):
    """自动检测CT最佳显示窗"""
    brain_tissue_range = (20, 100)
    hist, bins = np.histogram(img_data, bins=256, range=brain_tissue_range)
    peak = bins[np.argmax(hist)]

    if peak < 40:
        return 35, 110
    elif 40 <= peak < 60:
        return 40, 80
    else:
        return 80, 200


def auto_detect_mr_window(img_data):
    """自动检测MR显示范围"""
    p1 = np.percentile(img_data, 2)
    p99 = np.percentile(img_data, 98)
    return (p1 + p99) / 2, p99 - p1


def process_and_save_slice(args, save_dir, low, high, upscale_factor,
                           modality, output_format, jpeg_quality,
                           enable_clahe, original_spacing, blank_threshold):
    """处理单个切片（增强空白检测）"""
    try:
        slice_num, slice_data = args

        # 阶段0：原始数据全黑检测
        if np.max(slice_data) < 1:
            logger.debug(f"原始数据全黑切片 {slice_num}")
            return 0

        # 应用窗设置
        clipped = np.clip(slice_data, low, high)
        if np.all(clipped == 0):
            logger.debug(f"窗处理后全黑切片 {slice_num}")
            return 0

        # 增强处理
        enhanced = enhance_normalization(
            clipped,
            low=low,
            high=high,
            enable_clahe=enable_clahe,
            modality=modality
        )

        # 智能上采样
        upscaled = smart_upscale(
            enhanced,
            factor=upscale_factor,
            modality=modality,
            original_spacing=original_spacing
        )
        upscaled[upscaled < 1e-3] = 0

        # 最终全黑检测
        if is_blank_image(upscaled, blank_threshold):
            logger.debug(f"最终检测到空白切片 {slice_num}")
            return 0

        # 保存图像
        save_path = os.path.join(save_dir, f"{slice_num:04d}.{output_format}")
        save_image(upscaled, save_path, output_format, jpeg_quality)
        return 1

    except Exception as e:
        logger.warning(f"切片 {slice_num} 处理失败: {str(e)}")
        return 0


def enhance_normalization(slice_data, low, high, enable_clahe, modality):
    """增强处理"""
    normalized = np.interp(slice_data, [low, high], [0.0, 255.0]).astype(np.float32)

    if enable_clahe:
        normalized = apply_clahe(normalized, modality)

    gamma = 0.9 if modality == 'CT' else 1.1
    normalized = np.power(normalized / 255.0, gamma) * 255

    return np.uint8(np.clip(normalized, 0, 255))


def apply_clahe(img, modality):
    """对比度增强"""
    clahe_params = {
        'CT': {'clipLimit': 1.5, 'tileGridSize': (4, 4)},
        'MR': {'clipLimit': 2.0, 'tileGridSize': (8, 8)}
    }
    clahe = cv2.createCLAHE(**clahe_params[modality])
    return clahe.apply(np.uint8(img))


def smart_upscale(img, factor, modality, original_spacing):
    """智能上采样"""
    if original_spacing and len(original_spacing) == 3:
        z_ratio = original_spacing[2] / original_spacing[0]
        factor = int(factor * z_ratio)

    method = cv2.INTER_LANCZOS4 if modality == 'MR' else cv2.INTER_CUBIC
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=method)


def save_image(img, save_path, output_format, quality):
    """保存图像"""
    try:
        params = []
        if output_format.lower() in ('jpg', 'jpeg'):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]

        cv2.imwrite(save_path, img, params)
        logger.debug(f"已保存: {save_path}")
    except Exception as e:
        logger.error(f"保存失败 {save_path}: {str(e)}")
        raise


# 全黑图像检测模块
def remove_blank_images(output_dir, blank_threshold=0.01):
    """通用全黑图像清理"""
    logger.info(f"开始全黑图像清理，目录: {output_dir}")

    files = [f for f in os.listdir(output_dir)
             if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    logger.info(f"发现 {len(files)} 个待检查图像")

    removed_count = 0

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(
            check_and_remove,
            os.path.join(output_dir, f),
            blank_threshold
        ) for f in files]

        removed_count = sum(f.result() for f in futures)

    logger.info(f"清理完成，删除 {removed_count} 个空白图像")
    return removed_count


def remove_black_jpg(folder_path, blank_threshold=0.01):
    """JPG专项清理"""
    logger.info(f"开始JPG专项清理，目录: {folder_path}")

    jpg_files = []
    for f in os.listdir(folder_path):
        if f.lower().endswith(('.jpg', '.jpeg')):
            jpg_files.append(os.path.join(folder_path, f))
    logger.info(f"发现 {len(jpg_files)} 个JPG文件")

    removed_count = 0

    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        futures = [executor.submit(
            validate_and_remove_jpg,
            jpg_path,
            blank_threshold
        ) for jpg_path in jpg_files]

        removed_count = sum(f.result() for f in futures)

    logger.info(f"JPG清理完成，删除 {removed_count} 个无效文件")
    return removed_count


def check_and_remove(file_path, blank_threshold):
    """通用检测删除逻辑"""
    try:
        img = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR)
        if img is None:
            logger.warning(f"无法读取图像: {file_path}")
            return False

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if is_blank_image(img, blank_threshold):
            os.remove(file_path)
            logger.debug(f"删除空白图像: {os.path.basename(file_path)}")
            return True
        return False

    except Exception as e:
        logger.warning(f"处理 {file_path} 失败: {str(e)}")
        return False


def validate_and_remove_jpg(jpg_path, blank_threshold):
    """JPG专项验证逻辑"""
    try:
        # 快速元数据检查
        if os.path.getsize(jpg_path) < 1024:
            os.remove(jpg_path)
            logger.debug(f"删除小文件: {os.path.basename(jpg_path)}")
            return True

        img = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            os.remove(jpg_path)
            return True

        if is_black_jpg(img, blank_threshold):
            os.remove(jpg_path)
            logger.debug(f"删除空白JPG: {os.path.basename(jpg_path)}")
            return True
        return False

    except Exception as e:
        logger.error(f"处理 {jpg_path} 出错: {str(e)}")
        return False


def is_blank_image(img, threshold):
    """空白图像检测"""
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    total_pixels = img.size
    valid_pixels = total_pixels - hist[0][0]
    valid_ratio = valid_pixels / total_pixels
    return valid_ratio < threshold


def is_black_jpg(img, threshold):
    """JPG专用检测"""
    _, thresh = cv2.threshold(img, 5, 255, cv2.THRESH_TOZERO)
    h, w = img.shape
    roi = thresh[h // 4:h * 3 // 4, w // 4:w * 3 // 4]

    pixel_ratio = np.mean(roi > 10)
    intensity_ratio = np.mean(roi) / 255
    return pixel_ratio < threshold and intensity_ratio < threshold / 2


# 医学图像处理专用
def n4_bias_correction(image):
    """N4偏置场校正"""
    logger.info("执行N4偏置场校正...")
    image = sitk.Cast(image, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30])
    return corrector.Execute(image, mask)


if __name__ == "__main__":
    # CT处理示例
    mha_to_highres(
        save_dir='result/ct_brain',
        mha_path='./data/MambaMorph_Data/volumes_center/ct.nii.gz',
        modality='CT',
        output_format='jpg',
        remove_black_images=True
    )

    # MR处理示例
    mha_to_highres(
        save_dir='result/mr_t2',
        mha_path='./data/MambaMorph_Data/volumes_center/mr.nii.gz',
        modality='MR',
        correct_bias_field=True,
        output_format='png'
    )
