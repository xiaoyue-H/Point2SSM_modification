import os
import torch 
import torch.utils.data as data
import numpy as np
import pyvista as pv
import random
import trimesh
import matplotlib.pyplot as plt
import wandb
import plotly.tools as tls
import plotly.graph_objects as go

from utils.model_utils import calc_cd
import pointnet2_cuda as pointnet2

class MeshDataset(data.Dataset):
    def __init__(self, args, set_type='test', scale_factor=None):
        self.num_points = args.num_input_points
        self.sample_points = 16384
        # self.mesh_dir = os.path.join('data', args.dataset, set_type+'_meshes/')
        self.mesh_dir = os.path.join('/home/xiaoyue/shapenet', args.dataset)
        self.missing_percent = args.missing_percent
        self.noise_level = args.noise_level
        if self.noise_level== None or self.noise_level==0:
            self.add_noise = False
        else:
            self.add_noise = True
        self.subsample = args.train_subset_size
        self.set_type = set_type

        self.point_sets = []
        self.names = []
        
        calc_scale_factor = 0
        min_points = 1e8
        # for file in sorted(os.listdir(self.mesh_dir)):
        #     points = np.array(pv.read(self.mesh_dir+file).points)
        #     if np.max(np.abs(points)) > calc_scale_factor:
        #         calc_scale_factor = np.max(np.abs(points))
        #     if points.shape[0] < min_points:
        #         min_points = points.shape[0]
        #     self.point_sets.append(points)
        #     self.names.append(file.replace(".vtk",""))
        # self.min_points = min_points
        ############################################################################################
        for subdir in sorted(os.listdir(self.mesh_dir)):
            obj_path = os.path.join(self.mesh_dir, subdir, "models/model_normalized.obj")
            mesh = trimesh.load_mesh(obj_path)

            # Step 1: 在表面均匀采样 sample_points 个点
            points = mesh.sample(self.sample_points)
            # print(points.shape)
            # breakpoint()

            if points.shape[0] == 0:
                continue  # 跳过空文件

            points = np.array(points, dtype=np.float32)
            ############################################################################
            # 可视化点云
            if set_type == 'xiba':
                # 本来想用matplotlib形式，但我不知道为什么报错，所以先改用plotly(改回matplotlib静态图片尝试)
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')

                ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=self.normalize_coordinates_to_rgb(points), marker='o', s=100)  # 蓝色小点
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f'Point Cloud: {subdir}')
                x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
                y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
                z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

                max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0

                mid_x = (x_max + x_min) / 2.0
                mid_y = (y_max + y_min) / 2.0
                mid_z = (z_max + z_min) / 2.0

                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
                ax.set_box_aspect([1, 1, 1])

                # wandb.log({"Train Point Cloud": wandb.Image(fig, caption=subdir)})
                # plotly_fig = tls.mpl_to_plotly(fig)
                # wandb.log({"Train Point Cloud": wandb.Plotly(plotly_fig)})
                #
                plt.show()
                # plt.close(fig)

                # TODO plotly 似乎也不能正确显示
                # scatter = go.Scatter3d(
                #     x=points[:, 0], y=points[:, 1], z=points[:, 2],
                #     mode='markers',
                #     marker=dict(size=2, color='blue', opacity=0.8)  # 透明度 0.8 让点云更清晰
                # )
                #
                # # 创建交互式 Figure
                # plotly_fig = go.Figure(data=[scatter])
                # plotly_fig.update_layout(
                #     title=f'Point Cloud: {subdir}',
                #     scene=dict(
                #         xaxis_title="X",
                #         yaxis_title="Y",
                #         zaxis_title="Z",
                #         aspectmode='data'  # 保持 x/y/z 轴比例一致
                #     )
                # )
                #
                # # 记录到 wandb
                # wandb.log({"Train Point Cloud": wandb.Plotly(plotly_fig)})
                #
                # breakpoint()
            ############################################################################
            # 计算 scale_factor
            calc_scale_factor = max(calc_scale_factor, np.max(np.abs(points)))
            min_points = min(min_points, points.shape[0])

            self.point_sets.append(points)
            self.names.append(subdir)
        ##########################################################################################
        # 遍历 dataset 目录下的所有子目录
        # TODO 旧方法，只取 verticies 应当被弃用
        # for subdir in sorted(os.listdir(self.mesh_dir)):
        #     obj_path = os.path.join(self.mesh_dir, subdir, "models/model_normalized.obj")
        #     # print(f"path:{obj_path}")
        #     if not os.path.exists(obj_path):
        #         continue  # 跳过没有 obj 文件的子目录
        #
        #     points = self.load_obj(obj_path)  # 读取 .obj 文件
        #
        #     if points.shape[0] == 0:
        #         continue  # 跳过空文件
        #
        #     # 计算 scale_factor
        #     calc_scale_factor = max(calc_scale_factor, np.max(np.abs(points)))
        #     min_points = min(min_points, points.shape[0])
        #
        #     self.point_sets.append(points)
        #     self.names.append(subdir)  # 使用子目录名作为 name
        # print(f"✅ 读取到 {len(self.point_sets)} 个点云文件")
        ###########################################################################################
        self.min_points = min_points

        if not scale_factor:
            self.scale_factor = float(calc_scale_factor)
        else:
            self.scale_factor = scale_factor

        # if self.subsample != None and set_type=='train':
        #     if os.path.exists(self.mesh_dir + "../importance_sampling_indices.npy"):
        #         print("Using importance sampling.")
        #         sorted_indices = np.load(self.mesh_dir + "../importance_sampling_indices.npy")
        #         indices = sorted_indices[:int(self.subsample)]
        #         pts, nms = [], []
        #         for index in indices:
        #             pts.append(self.point_sets[index])
        #             nms.append(self.names[index])
        #     else:
        #         pts, nms = self.point_sets[:int(self.subsample)], self.names[:int(self.subsample)]
        #     self.point_sets = pts
        #     self.names = nms

    def normalize_coordinates_to_rgb(self, data):
        # 归一化处理，保证每个点的坐标都在 [0, 1] 范围内
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        normalized = (data - min_vals) / (max_vals - min_vals)
        # 将归一化后的数据映射为 RGB 值
        return normalized

    def load_obj(self, file_path):
        """ 读取 .obj 文件中的点云数据 """
        points = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue  # 忽略格式不正确的行

                    if parts[0] == 'v':  # 只读取顶点数据
                        try:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            points.append([x, y, z])
                        except ValueError:
                            print(f"⚠️ 无效的点数据: {line}")
                            continue

            points = np.array(points, dtype=np.float32)
            # 可视化点云
            # fig = plt.figure(figsize=(8, 6))
            # ax = fig.add_subplot(111, projection='3d')
            #
            # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', s=1)  # 蓝色小点
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            # ax.set_title('Point Cloud Visualization (Matplotlib)')
            #
            # plt.show()
            return np.array(points)
        except Exception as e:
            print(f"❌ 读取 {file_path} 失败: {e}")
            return None


    def get_scale_factor(self):
        return self.scale_factor

    def __getitem__(self, index):
        full_point_set = self.point_sets[index]
        name = self.names[index]
        
        # add missingness
        if not self.missing_percent or self.missing_percent == 0:
            partial_point_set = full_point_set
        else:
            if self.set_type == 'train':
                seed = np.random.randint(len(full_point_set))
            else:
                seed = 0 # consistent testing
            distances = np.linalg.norm(full_point_set - full_point_set[seed], axis=1)
            sorted_points = full_point_set[np.argsort(distances)]
            partial_point_set = sorted_points[int(len(full_point_set)*self.missing_percent):]

        # select subset
        if self.num_points > len(partial_point_set):
            replace = True
        else: 
            replace = False
        choice = np.random.choice(len(partial_point_set), self.num_points, replace=replace)
        partial = torch.FloatTensor(partial_point_set[choice, :])
        
        # add noise
        if self.add_noise:
            partial = partial + (self.noise_level)*torch.randn(partial.shape)
        
        # ground truth 
        choice = np.random.choice(len(full_point_set), self.min_points, replace=False)
        gt = torch.FloatTensor(full_point_set[choice, :])
        
        return partial/self.scale_factor, gt/self.scale_factor, name

    def __len__(self):
        return len(self.point_sets)

'''
If ref path is none it will use a random refs
'''
class DPC_Dataset(data.Dataset):
    def __init__(self, args, set_type='test', scale_factor=None, ref_path=None):
        self.num_points = args.num_input_points
        self.mesh_dataset = MeshDataset(args, set_type, scale_factor)
        self.scale_factor = self.mesh_dataset.scale_factor
        if ref_path:
            ref_points = np.array(pv.read(ref_path).points)
            target_pc = torch.FloatTensor(ref_points/ self.scale_factor).to('cuda:0')
            self.target_pc = furthest_point_downsampling(target_pc[None,:], self.num_points).squeeze()
        else:
            self.target_pc = None
            
    def get_scale_factor(self):
        return self.scale_factor
        
    def __getitem__(self, index):
        source_pc, source_gt, source_name = self.mesh_dataset.__getitem__(index)
        if self.target_pc == None:
            choices = list(range(0,index)) + list(range(index+1, len(self.mesh_dataset.point_sets)))
            target_index = random.choice(choices)
            target_pc, target_gt, target_name = self.mesh_dataset.__getitem__(target_index)
        else:
            target_pc = self.target_pc
        return source_pc, target_pc, source_gt, source_name

    def __len__(self):
        return len(self.mesh_dataset.point_sets)


def furthest_point_downsampling(points, npoint):
    xyz = points.contiguous()
    B, N, _ = xyz.size()
    output = torch.IntTensor(B, npoint).to(xyz.device)
    temp = torch.FloatTensor(B, N).fill_(1e10).to(xyz.device)
    pointnet2.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
    indices = output.cpu().numpy()
    subset = []
    for i in range(points.shape[0]):
        subset.append(points[i][indices[i], :].cpu().numpy())
    subset = np.array(subset)
    return torch.FloatTensor(subset).to('cuda:0')
