"""
方向盘皮料缺陷检测系统 - 任务三（修复版）
"""

import cv2
import numpy as np
import os
from pathlib import Path
import time

class LeatherDefectDetector:
    """皮料缺陷检测器 - 高特异性版本"""
    
    def __init__(self, config=None):
        """初始化检测器参数"""
        self.config = {
            # 预处理参数
            'gaussian_kernel': (5, 5),
            'gaussian_sigma': 1.5,
            'clahe_clip_limit': 1.5,
            'clahe_grid_size': (8, 8),
            
            # 策略一：灰度阈值分割（白漆、高亮缺陷）
            'min_area_highlight': 150,
            'max_area_highlight': 5000,
            'intensity_threshold': 210,
            
            # 策略二：纹理分析（褶皱）
            'texture_window_size': 25,
            'texture_variance_threshold': 4.0,
            'min_area_texture': 150,
            'max_area_texture': 15000,  # 添加缺失的参数
            
            # 策略三：边缘梯度检测（划痕）
            'gradient_threshold': 70,
            'min_area_gradient': 80,
            'max_area_gradient': 10000,  # 添加缺失的参数
            
            # 策略四：暗斑检测（凹坑）
            'dark_spot_threshold': 40,
            'min_area_dark': 40,
            'max_area_dark': 400,
            
            # 融合策略
            'fusion_method': 'voting',
            'voting_threshold': 2,
            
            # 形态学核
            'morphology_kernel': (3, 3),
        }
        
        if config:
            self.config.update(config)
        
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            self.config['morphology_kernel']
        )
    
    def preprocess(self, image):
        """图像预处理"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        blurred = cv2.GaussianBlur(gray, self.config['gaussian_kernel'], self.config['gaussian_sigma'])
        
        clahe = cv2.createCLAHE(
            clipLimit=self.config['clahe_clip_limit'],
            tileGridSize=self.config['clahe_grid_size']
        )
        enhanced = clahe.apply(blurred)
        
        return gray, enhanced
    
    def detect_by_intensity(self, enhanced_img, original_gray):
        """策略一：基于灰度强度的缺陷检测"""
        _, mask = cv2.threshold(enhanced_img, self.config['intensity_threshold'], 255, cv2.THRESH_BINARY)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config['min_area_highlight'] <= area <= self.config['max_area_highlight']:
                x, y, w, h = cv2.boundingRect(contour)
                roi = enhanced_img[y:y+h, x:x+w]
                mean_brightness = np.mean(roi)
                if mean_brightness > self.config['intensity_threshold']:
                    cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
        
        score = min(1.0, np.sum(filtered_mask) / 5000) if np.sum(filtered_mask) > 0 else 0.0
        return filtered_mask, score
    
    def detect_by_texture(self, original_gray):
        """策略二：基于局部方差的纹理分析"""
        h, w = original_gray.shape
        window_size = self.config['texture_window_size']
        half = window_size // 2
        
        variance_map = np.zeros_like(original_gray, dtype=np.float32)
        
        integral = cv2.integral(original_gray.astype(np.float32))
        integral_sq = cv2.integral((original_gray.astype(np.float32) ** 2))
        
        for i in range(half, h - half):
            for j in range(half, w - half):
                x1, y1 = i - half, j - half
                x2, y2 = i + half + 1, j + half + 1
                
                sum_pixel = integral[x2, y2] - integral[x1, y2] - integral[x2, y1] + integral[x1, y1]
                sum_sq = integral_sq[x2, y2] - integral_sq[x1, y2] - integral_sq[x2, y1] + integral_sq[x1, y1]
                
                n = window_size ** 2
                mean = sum_pixel / n
                variance = max(0, (sum_sq / n) - (mean ** 2))
                variance_map[i, j] = variance
        
        mean_var = np.mean(variance_map)
        std_var = np.std(variance_map)
        threshold = mean_var + self.config['texture_variance_threshold'] * std_var
        
        mask = (variance_map > threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config['min_area_texture'] <= area <= self.config['max_area_texture']:
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
        
        score = min(1.0, np.sum(filtered_mask) / 10000) if np.sum(filtered_mask) > 0 else 0.0
        return filtered_mask, score
    
    def detect_by_gradient(self, original_gray):
        """策略三：基于边缘梯度的缺陷检测"""
        sobel_x = cv2.Sobel(original_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(original_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient = np.uint8(np.clip(gradient / gradient.max() * 255, 0, 255))
        
        _, mask = cv2.threshold(gradient, self.config['gradient_threshold'], 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config['min_area_gradient'] <= area:
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
        
        score = min(1.0, np.sum(filtered_mask) / 5000) if np.sum(filtered_mask) > 0 else 0.0
        return filtered_mask, score
    
    def detect_by_dark_spot(self, original_gray):
        """策略四：暗斑检测"""
        _, mask = cv2.threshold(original_gray, self.config['dark_spot_threshold'], 255, cv2.THRESH_BINARY_INV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config['min_area_dark'] <= area <= self.config['max_area_dark']:
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
        
        score = min(1.0, np.sum(filtered_mask) / 2000) if np.sum(filtered_mask) > 0 else 0.0
        return filtered_mask, score
    
    def fuse_results(self, masks, scores, method='voting'):
        """融合多个检测策略的结果"""
        if len(masks) == 0:
            return np.zeros((100, 100), dtype=np.uint8), 0.0
        
        h, w = masks[0][0].shape
        
        if method == 'or':
            final_mask = np.zeros((h, w), dtype=np.uint8)
            for mask, _ in masks:
                final_mask = cv2.bitwise_or(final_mask, mask)
            final_score = max(scores) if scores else 0.0
            
        elif method == 'voting':
            vote_mask = np.zeros((h, w), dtype=np.uint8)
            for mask, _ in masks:
                vote_mask += (mask > 0).astype(np.uint8)
            final_mask = (vote_mask >= self.config['voting_threshold']).astype(np.uint8) * 255
            final_score = np.mean(scores) if scores else 0.0
        
        return final_mask, final_score
    
    def detect(self, image):
        """主检测函数"""
        original_gray, enhanced = self.preprocess(image)
        
        intensity_mask, intensity_score = self.detect_by_intensity(enhanced, original_gray)
        texture_mask, texture_score = self.detect_by_texture(original_gray)
        gradient_mask, gradient_score = self.detect_by_gradient(original_gray)
        dark_mask, dark_score = self.detect_by_dark_spot(original_gray)
        
        masks = [
            (intensity_mask, intensity_score),
            (texture_mask, texture_score),
            (gradient_mask, gradient_score),
            (dark_mask, dark_score)
        ]
        scores = [intensity_score, texture_score, gradient_score, dark_score]
        
        final_mask, final_score = self.fuse_results(masks, scores, method=self.config['fusion_method'])
        
        has_defect = np.sum(final_mask) > 0
        
        result_image = image.copy()
        if has_defect:
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)
        
        return {
            'has_defect': has_defect,
            'defect_mask': final_mask,
            'confidence': final_score,
            'scores': {
                'intensity': intensity_score,
                'texture': texture_score,
                'gradient': gradient_score,
                'dark_spot': dark_score
            },
            'result_image': result_image
        }


class DatasetEvaluator:
    """数据集评估器"""
    
    def __init__(self, detector):
        self.detector = detector
    
    def scan_dataset(self, data_root):
        """扫描数据集结构"""
        data_root = Path(data_root)
        
        golden_dir = data_root / 'golden_dataset'
        imperfect_dir = data_root / 'imperfect_dataset'
        defect_image_dir = imperfect_dir / 'defect_image'
        
        golden_images = []
        if golden_dir.exists():
            golden_images = [str(p) for p in golden_dir.glob('*') 
                           if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']]
        
        defect_images = []
        if defect_image_dir.exists():
            defect_images = [str(p) for p in defect_image_dir.glob('*')
                           if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']]
        
        print(f"\n数据集扫描结果:")
        print(f"  golden_dataset (无瑕疵): {len(golden_images)} 张")
        print(f"  imperfect_dataset/defect_image (有瑕疵): {len(defect_images)} 张")
        
        return golden_images, defect_images
    
    def evaluate(self, data_root, verbose=True, save_results=True):
        """评估检测性能"""
        golden_images, defect_images = self.scan_dataset(data_root)
        
        true_defect_images = defect_images
        true_normal_images = golden_images
        
        if verbose:
            print(f"\n评估数据集划分:")
            print(f"  真实有缺陷样本: {len(true_defect_images)}")
            print(f"  真实无缺陷样本: {len(true_normal_images)}")
        
        tp, fn, fp, tn = 0, 0, 0, 0
        detection_times = []
        fp_list = []
        fn_list = []
        
        if save_results:
            result_dir = Path(data_root) / 'detection_results'
            result_dir.mkdir(exist_ok=True)
        
        # 检测有缺陷样本
        if verbose:
            print("\n" + "="*60)
            print("检测有缺陷样本:")
            print("="*60)
        
        for img_path in true_defect_images:
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            start_time = time.time()
            result = self.detector.detect(image)
            detection_times.append(time.time() - start_time)
            
            img_name = os.path.basename(img_path)
            
            if result['has_defect']:
                tp += 1
                status = "✓ TP"
            else:
                fn += 1
                status = "✗ FN"
                fn_list.append(img_name)
            
            if verbose:
                s = result['scores']
                print(f"  [{status}] {img_name:35s} | 强度:{s['intensity']:.1f} 纹理:{s['texture']:.1f} 梯度:{s['gradient']:.1f} 暗斑:{s['dark_spot']:.1f}")
            
            if save_results:
                save_path = result_dir / f"result_{img_name}"
                cv2.imwrite(str(save_path), result['result_image'])
        
        # 检测无缺陷样本
        if verbose:
            print("\n" + "="*60)
            print("检测无缺陷样本:")
            print("="*60)
        
        for img_path in true_normal_images:
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            start_time = time.time()
            result = self.detector.detect(image)
            detection_times.append(time.time() - start_time)
            
            img_name = os.path.basename(img_path)
            
            if result['has_defect']:
                fp += 1
                status = "✗ FP"
                fp_list.append(img_name)
            else:
                tn += 1
                status = "✓ TN"
            
            if verbose and result['has_defect']:
                s = result['scores']
                print(f"  [{status}] {img_name:35s} | 强度:{s['intensity']:.1f} 纹理:{s['texture']:.1f} 梯度:{s['gradient']:.1f} 暗斑:{s['dark_spot']:.1f}")
        
        # 计算指标
        miss_rate = fn / (tp + fn) * 100 if (tp + fn) > 0 else 0
        false_alarm_rate = fp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100 if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_time = np.mean(detection_times) if detection_times else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        metrics = {
            'true_positive': tp,
            'false_negative': fn,
            'false_positive': fp,
            'true_negative': tn,
            'miss_rate': miss_rate,
            'false_alarm_rate': false_alarm_rate,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_time_ms': avg_time * 1000,
            'fps': fps,
        }
        
        if verbose:
            print("\n" + "="*60)
            print("评估结果")
            print("="*60)
            print(f"\n  混淆矩阵:")
            print(f"               预测缺陷    预测正常")
            print(f"  真实缺陷      {tp:4d}          {fn:4d}")
            print(f"  真实正常      {fp:4d}          {tn:4d}")
            
            print(f"\n  性能指标:")
            print(f"    漏警率:     {miss_rate:.2f}%  (漏检 {fn} 张)")
            print(f"    虚警率:     {false_alarm_rate:.2f}%  (误报 {fp} 张)")
            print(f"    准确率:     {accuracy:.2f}%")
            print(f"    精确率:     {precision:.2f}%")
            print(f"    召回率:     {recall:.2f}%")
            print(f"    F1分数:     {f1_score:.2f}%")
            print(f"    平均时间:   {avg_time*1000:.2f}ms")
            print(f"    FPS:        {fps:.2f}")
            
            if fn_list:
                print(f"\n  漏检案例 ({len(fn_list)} 张):")
                for name in fn_list:
                    print(f"    - {name}")
            
            if fp_list:
                print(f"\n  虚警案例 ({len(fp_list)} 张):")
                for name in fp_list[:20]:
                    print(f"    - {name}")
                if len(fp_list) > 20:
                    print(f"    ... 共{len(fp_list)}张")
            
            def get_score(rate):
                if rate <= 1:
                    return 20
                elif rate <= 3:
                    return 18
                elif rate <= 5:
                    return 16
                elif rate <= 8:
                    return 14
                elif rate <= 10:
                    return 12
                elif rate <= 15:
                    return 10
                elif rate <= 20:
                    return 9
                else:
                    return 8
            
            miss_score = get_score(miss_rate)
            fa_score = get_score(false_alarm_rate)
            
            print(f"\n{'='*60}")
            print("项目评分（根据评分标准表5、表6）:")
            print(f"  漏警率 {miss_rate:.1f}% → {miss_score}分")
            print(f"  虚警率 {false_alarm_rate:.1f}% → {fa_score}分")
            print(f"  总分: {miss_score + fa_score}分")
            print(f"{'='*60}")
        
        return metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='方向盘皮料缺陷检测 - 任务三')
    parser.add_argument('--data_path', type=str, default='./database_3',
                       help='dataset3根目录路径')
    parser.add_argument('--test_image', type=str, default=None,
                       help='单张图像测试路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"错误: 数据集路径不存在: {args.data_path}")
        return
    
    # 高特异性配置
    config = {
        'min_area_highlight': 150,
        'max_area_highlight': 5000,
        'intensity_threshold': 210,
        'texture_window_size': 25,
        'texture_variance_threshold': 4,
        'min_area_texture': 150,
        'max_area_texture': 15000,
        'gradient_threshold': 70,
        'min_area_gradient': 80,
        'dark_spot_threshold': 40,
        'min_area_dark': 40,
        'max_area_dark': 400,
        'voting_threshold': 2,
    }
    
    detector = LeatherDefectDetector(config)
    evaluator = DatasetEvaluator(detector)
    
    if args.test_image:
        print(f"测试单张图像: {args.test_image}")
        image = cv2.imread(args.test_image)
        if image is not None:
            result = detector.detect(image)
            print(f"检测结果: {'有缺陷' if result['has_defect'] else '无缺陷'}")
            print(f"各策略得分: {result['scores']}")
        return
    
    print("="*60)
    print("方向盘皮料缺陷检测系统 - 任务三")
    print("="*60)
    print(f"数据集路径: {args.data_path}")
    
    metrics = evaluator.evaluate(args.data_path, verbose=True, save_results=True)


if __name__ == '__main__':
    main()