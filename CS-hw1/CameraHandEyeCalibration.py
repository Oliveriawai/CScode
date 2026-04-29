"""
CameraHandEyeCalibration.py - 相机标定+手眼标定
适用于OpenCV 4.9.0
"""

import cv2
import numpy as np
import glob
import os
import re

class CameraHandEyeCalibration:
    def __init__(self):
        # 棋盘格参数
        self.board_size = (9, 6)  # 内角点数：9列，6行
        self.square_size = 0.02   # 方格边长：0.02米
        
        # 标定结果存储
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.h_board2cam_list = []
        self.valid_image_indices = []
        
    def natural_sort_key(self, filename):
        """自然排序：提取数字进行排序"""
        basename = os.path.basename(filename)
        numbers = re.findall(r'\d+', basename)
        return int(numbers[0]) if numbers else 0
    
    def get_sorted_images(self, image_folder):
        """按数字顺序获取图片路径"""
        all_images = glob.glob(os.path.join(image_folder, "*.bmp"))
        all_images.sort(key=self.natural_sort_key)
        
        print(f"找到 {len(all_images)} 张图片（已按数字排序）:")
        for img in all_images:
            print(f"  {os.path.basename(img)}")
        
        return all_images
    
    def generate_object_points(self):
        """生成世界坐标系中的标定板角点坐标"""
        objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 
                               0:self.board_size[1]].T.reshape(-1, 2)
        objp[:, :2] = objp[:, :2] * self.square_size
        return objp
    
    def extract_corners(self, image_path, show_visualization=False):
        """提取棋盘格角点"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"  ✗ 无法读取图片")
            return False, None, None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(
            gray, 
            self.board_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            if show_visualization:
                img_copy = img.copy()
                cv2.drawChessboardCorners(img_copy, self.board_size, corners_sub, ret)
                cv2.imshow('Chessboard Corners', img_copy)
                cv2.waitKey(100)
            
            return True, corners_sub, img.shape[:2][::-1]
        
        return False, None, None
    
    def calibrate_camera(self, image_folder):
        """执行相机标定"""
        print("=" * 60)
        print("开始相机标定...")
        print("=" * 60)
        
        obj_points = []
        img_points = []
        objp = self.generate_object_points()
        
        image_paths = self.get_sorted_images(image_folder)
        
        if len(image_paths) == 0:
            print("\n错误：没有找到任何bmp图片！")
            return False
        
        print(f"\n开始处理 {len(image_paths)} 张图片...")
        print("-" * 40)
        
        img_size = None
        
        for idx, img_path in enumerate(image_paths, 1):
            print(f"处理图片 {idx}: {os.path.basename(img_path)}")
            ret, corners, size = self.extract_corners(img_path, show_visualization=True)
            
            if ret:
                obj_points.append(objp)
                img_points.append(corners)
                self.valid_image_indices.append(idx)
                if img_size is None:
                    img_size = size
                print(f"  ✓ 成功！找到 {len(corners)} 个角点")
            else:
                print(f"  ✗ 失败！未检测到棋盘格角点")
        
        cv2.destroyAllWindows()
        
        print("-" * 40)
        print(f"\n有效图片: {len(obj_points)} / {len(image_paths)}")
        
        if len(obj_points) < 3:
            print("错误：有效图片不足3张，无法标定！")
            return False
        
        # 执行标定
        print("\n正在计算相机参数...")
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_size, None, None
        )
        
        if ret:
            print("\n✓✓✓ 相机标定成功！✓✓✓")
            print("\n内参矩阵 K:")
            print(self.camera_matrix)
            print(f"\n焦距: fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}")
            print(f"主点: cx={self.camera_matrix[0,2]:.2f}, cy={self.camera_matrix[1,2]:.2f}")
            print(f"\n畸变系数: {self.dist_coeffs.flatten()}")
            
            # 计算重投影误差
            total_error = 0
            for i in range(len(obj_points)):
                imgpoints2, _ = cv2.projectPoints(obj_points[i], self.rvecs[i], self.tvecs[i],
                                                   self.camera_matrix, self.dist_coeffs)
                error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error
            mean_error = total_error / len(obj_points)
            print(f"\n平均重投影误差: {mean_error:.6f} 像素")
            
            # 构建外参矩阵
            self.build_board2cam_matrices()
            
            # 保存结果
            self.save_calibration_results()
            
            return True
        
        print("相机标定失败！")
        return False
    
    def build_board2cam_matrices(self):
        """构建标定板到相机的变换矩阵"""
        for i in range(len(self.rvecs)):
            R, _ = cv2.Rodrigues(self.rvecs[i])
            t = self.tvecs[i].reshape(3, 1)
            H = np.eye(4)
            H[:3, :3] = R
            H[:3, 3] = t.flatten()
            self.h_board2cam_list.append(H)
        
        print(f"\n已构建 {len(self.h_board2cam_list)} 个外参矩阵")
        print(f"对应图片序号: {self.valid_image_indices}")
    
    def save_calibration_results(self):
        """保存标定结果"""
        filename = "camera_calibration_results.txt"
        with open(filename, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("相机标定结果\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("棋盘格参数:\n")
            f.write(f"  内角点数: {self.board_size[0]} x {self.board_size[1]}\n")
            f.write(f"  方格边长: {self.square_size} 米\n\n")
            
            f.write("内参矩阵 K:\n")
            np.savetxt(f, self.camera_matrix, fmt='%.8f')
            f.write("\n")
            
            f.write("畸变系数:\n")
            np.savetxt(f, self.dist_coeffs.flatten().reshape(1, -1), fmt='%.8f')
            f.write("\n")
            
            f.write(f"平均重投影误差: \n\n")
            f.write(f"有效图片序号: {self.valid_image_indices}\n\n")
            
            f.write("每张图像的外参矩阵 (标定板→相机):\n")
            for i, H in enumerate(self.h_board2cam_list):
                f.write(f"\n图片 {self.valid_image_indices[i]}:\n")
                np.savetxt(f, H, fmt='%.8f')
        
        print(f"\n标定结果已保存至: {filename}")
    
    def parse_robot_poses(self, pose_file):
        """解析机械臂位姿文件（支持带文字说明的格式）"""
        print("\n" + "=" * 60)
        print("加载机械臂位姿...")
        print("=" * 60)
        
        if not os.path.exists(pose_file):
            print(f"错误：找不到文件 {pose_file}")
            return []
        
        poses = []
        
        with open(pose_file, 'r') as f:
            lines = f.readlines()
        
        print(f"读取到 {len(lines)} 行数据")
        
        current_pose = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 跳过包含文字的标题行
            if line.startswith("Robot Pose") or line.startswith("Hand-eye"):
                if len(current_pose) == 4:
                    poses.append(np.array(current_pose))
                    current_pose = []
                continue
            
            # 尝试解析数字行
            parts = line.split()
            if len(parts) == 4:
                try:
                    row = [float(x) for x in parts]
                    current_pose.append(row)
                except ValueError:
                    continue
            
            # 如果收集够4行，保存位姿
            if len(current_pose) == 4:
                poses.append(np.array(current_pose))
                current_pose = []
        
        # 检查最后是否有未保存的位姿
        if len(current_pose) == 4:
            poses.append(np.array(current_pose))
        
        print(f"成功解析 {len(poses)} 个位姿矩阵")
        
        # 显示前3个位姿用于验证
        for i, pose in enumerate(poses[:3]):
            print(f"\n位姿 {i+1} (对应图片 {i+1}):")
            print(pose)
        
        return poses
    
    def hand_eye_calibration(self, robot_pose_file):
        """执行手眼标定"""
        print("\n" + "=" * 60)
        print("开始手眼标定...")
        print("=" * 60)
        
        robot_poses = self.parse_robot_poses(robot_pose_file)
        
        if len(robot_poses) == 0:
            print("错误：没有有效的机械臂位姿数据！")
            return None
        
        # 确保数量匹配（使用有效图片对应的数量）
        n_effective = min(len(self.h_board2cam_list), len(robot_poses))
        
        print(f"\n相机外参数量: {len(self.h_board2cam_list)}")
        print(f"机械臂位姿数量: {len(robot_poses)}")
        print(f"将使用前 {n_effective} 组数据进行手眼标定")
        
        if n_effective < 2:
            print(f"错误：有效数据不足！需要至少2组数据")
            return None
        
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []
        
        for i in range(n_effective):
            H_gripper2base = robot_poses[i]
            R_gripper2base.append(H_gripper2base[:3, :3])
            t_gripper2base.append(H_gripper2base[:3, 3])
            
            H_board2cam = self.h_board2cam_list[i]
            R_target2cam.append(H_board2cam[:3, :3])
            t_target2cam.append(H_board2cam[:3, 3])
        
        print("\n正在求解 AX = XB 方程...")
        
        results = {}
        
        # 尝试多种方法（只使用OpenCV 4.9.0支持的方法）
        methods = [
            (cv2.CALIB_HAND_EYE_TSAI, "Tsai (经典方法)"),
            (cv2.CALIB_HAND_EYE_PARK, "Park (基于李群理论)")
        ]
        
        for method, name in methods:
            try:
                R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam,
                    method=method
                )
                
                H_cam2gripper = np.eye(4)
                H_cam2gripper[:3, :3] = R_cam2gripper
                H_cam2gripper[:3, 3] = t_cam2gripper.flatten()
                
                results[name] = H_cam2gripper
                
                print(f"\n{'='*50}")
                print(f"{name} 方法结果:")
                print(f"{'='*50}")
                print("相机→机械臂末端变换矩阵 H_cam2gripper:")
                print(H_cam2gripper)
                print(f"\n旋转矩阵 R:")
                print(R_cam2gripper)
                print(f"\n平移向量 t: {t_cam2gripper.flatten()}")
                
            except Exception as e:
                print(f"{name} 方法失败: {e}")
        
        if results:
            # 选择第一种方法的结果
            best_name = list(results.keys())[0]
            best_result = results[best_name]
            
            print("\n" + "=" * 60)
            print(f"✓✓✓ 手眼标定成功！(使用 {best_name}) ✓✓✓")
            print("=" * 60)
            
            # 验证结果
            self.verify_result(robot_poses, best_result, n_effective)
            
            # 保存结果
            self.save_handeye_results(best_result, best_name)
            
            return best_result
        
        return None
    
    def verify_result(self, robot_poses, H_cam2gripper, n_effective):
        """验证手眼标定结果"""
        print("\n" + "=" * 60)
        print("验证 AX = XB 方程")
        print("=" * 60)
        
        errors = []
        for i in range(min(5, n_effective - 1)):
            H_gripper2base1 = robot_poses[i]
            H_gripper2base2 = robot_poses[i + 1]
            H_board2cam1 = self.h_board2cam_list[i]
            H_board2cam2 = self.h_board2cam_list[i + 1]
            
            # A: 机械臂运动
            A = np.linalg.inv(H_gripper2base1) @ H_gripper2base2
            # B: 标定板运动
            B = H_board2cam1 @ np.linalg.inv(H_board2cam2)
            
            # 验证 AX = XB
            AX = A @ H_cam2gripper
            XB = H_cam2gripper @ B
            
            error = np.linalg.norm(AX - XB)
            errors.append(error)
            print(f"第{i+1}组 (图片{self.valid_image_indices[i]}→{self.valid_image_indices[i+1]}): 误差 = {error:.8f}")
        
        print(f"\n平均验证误差: {np.mean(errors):.8f}")
        
        if np.mean(errors) < 0.01:
            print("✓ 验证通过！")
        else:
            print("⚠ 误差较大，请检查数据")
    
    def save_handeye_results(self, H_cam2gripper, method):
        """保存手眼标定结果"""
        filename = "handeye_calibration_results.txt"
        with open(filename, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("手眼标定结果\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"使用算法: {method}\n")
            f.write(f"有效图片数量: {len(self.h_board2cam_list)}\n")
            f.write(f"图片序号: {self.valid_image_indices}\n\n")
            
            f.write("相机→机械臂末端变换矩阵 H_cam2gripper:\n")
            np.savetxt(f, H_cam2gripper, fmt='%.8f')
            f.write("\n")
            
            f.write("旋转矩阵 R:\n")
            np.savetxt(f, H_cam2gripper[:3, :3], fmt='%.8f')
            f.write("\n")
            
            f.write(f"平移向量 t: {H_cam2gripper[:3, 3]}\n")
            
            # 计算欧拉角
            R = H_cam2gripper[:3, :3]
            sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
            if sy > 1e-6:
                roll = np.arctan2(R[2,1], R[2,2])
                pitch = np.arctan2(-R[2,0], sy)
                yaw = np.arctan2(R[1,0], R[0,0])
            else:
                roll = np.arctan2(-R[1,2], R[1,1])
                pitch = np.arctan2(-R[2,0], sy)
                yaw = 0
            
            f.write(f"\n欧拉角 (度): roll={np.degrees(roll):.2f}°, pitch={np.degrees(pitch):.2f}°, yaw={np.degrees(yaw):.2f}°\n")
        
        print(f"\n手眼标定结果已保存至: {filename}")


def main():
    print("=" * 60)
    print("相机标定 + 手眼标定程序")
    print("=" * 60)
    print(f"当前工作目录: {os.getcwd()}")
    print()
    
    # 检查必要文件
    if not os.path.exists("calib_images"):
        print("错误：找不到 calib_images 文件夹！")
        return
    
    # 创建标定器并运行
    calibrator = CameraHandEyeCalibration()
    
    # 相机标定
    success = calibrator.calibrate_camera("calib_images")
    
    if not success:
        print("\n相机标定失败！")
        return
    
    # 手眼标定
    if os.path.exists("robot_poses.txt"):
        H = calibrator.hand_eye_calibration("robot_poses.txt")
        if H is not None:
            print("\n" + "=" * 60)
            print("实验全部完成！")
            print("=" * 60)
            
            # 输出最终结果摘要
            print("\n最终结果摘要:")
            print(f"相机内参 fx={calibrator.camera_matrix[0,0]:.2f}, fy={calibrator.camera_matrix[1,1]:.2f}")
            print(f"相机主点 cx={calibrator.camera_matrix[0,2]:.2f}, cy={calibrator.camera_matrix[1,2]:.2f}")
            print(f"\n手眼变换矩阵 H_cam2gripper:")
            print(H)
    else:
        print("\n跳过手眼标定（缺少robot_poses.txt）")


if __name__ == "__main__":
    main()