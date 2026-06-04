import os
import mujoco
import mujoco.viewer
import numpy as np
import time
from scipy import ndimage

def generate_rich_environment():
    """
    生成丰富的地形环境，包含缩短的正弦沟壑和平地区域
    每次运行时沟壑参数随机生成，增加多样性
    """
    nrow, ncol = 257, 257
    size_x, size_y = 12.0, 12.0
    base_height = 0.6
    
    # 创建网格
    x = np.linspace(-size_x, size_x, ncol)
    y = np.linspace(-size_y, size_y, nrow)
    X, Y = np.meshgrid(x, y)

    # 初始化平坦地形
    terrain = np.ones((nrow, ncol)) * base_height

    # 定义沟壑的x轴范围
    trench_start_x = -6.0
    trench_end_x = 6.0

    # 随机生成沟壑参数
    trench_depth = np.random.uniform(0.8, 1.0)
    trench_width = np.random.uniform(9.0, 12.0)
    amplitude = np.random.uniform(1.5, 2.2)
    wavelength = np.random.uniform(8.0, 10.0)

    print(f"随机沟壑参数 - 深度: {trench_depth:.2f}, 宽度: {trench_width:.2f}, "
          f"幅度: {amplitude:.2f}, 波长: {wavelength:.2f}")

    # 保存沟壑参数
    trench_params = {
        'trench_start_x': trench_start_x,
        'trench_end_x': trench_end_x,
        'trench_depth': trench_depth,
        'trench_width': trench_width,
        'amplitude': amplitude,
        'wavelength': wavelength
    }

    # 1. 创建缩短的正弦贯穿沟壑
    for i in range(ncol):
        # 检查当前x坐标是否在沟壑范围内
        if trench_start_x <= x[i] <= trench_end_x:
            # 计算当前X位置的正弦波Y值作为沟壑中心
            center_y = amplitude * np.sin(2 * np.pi * x[i] / wavelength)

            for j in range(nrow):
                # 计算当前点到正弦波中心的距离
                distance_to_center = abs(y[j] - center_y)

                # 如果点在沟壑宽度内，降低高度
                if distance_to_center < trench_width / 2:
                    # 使用平滑的过渡（高斯函数）
                    trench_factor = np.exp(-(distance_to_center ** 2) / (trench_width / 6) ** 2)
                    terrain[j, i] = base_height - trench_depth * trench_factor
        else:
            # 在沟壑范围外，保持平坦地形，高度设为沟壑底部高度
            terrain[:, i] = base_height - trench_depth

    # 2. 添加丘陵地形（只添加在沟壑范围内）
    # 定义沟壑区域，避免丘陵与沟壑重叠
    trench_region = np.zeros((nrow, ncol), dtype=bool)
    for i in range(ncol):
        if trench_start_x <= x[i] <= trench_end_x:
            center_y = amplitude * np.sin(2 * np.pi * x[i] / wavelength)
            for j in range(nrow):
                distance_to_center = abs(y[j] - center_y)
                if distance_to_center < trench_width * 1.5:  # 扩大沟壑区域
                    trench_region[j, i] = True

    # 添加固定丘陵
    hills = [
        (-8.0, 8.0, 3.0, 0.8),   # 左上角
        (8.0, 8.0, 2.5, 0.7),    # 右上角
        (-8.0, -8.0, 3.0, 0.6),  # 左下角
        (8.0, -8.0, 2.8, 0.7)    # 右下角
    ]

    for hill_center_x, hill_center_y, hill_radius, hill_height in hills:
        distance_to_hill = np.sqrt((X - hill_center_x) ** 2 + (Y - hill_center_y) ** 2)
        in_trench_range = (trench_start_x <= X) & (X <= trench_end_x)
        hill_mask = (distance_to_hill < hill_radius) & (~trench_region) & in_trench_range
        hill_factor = np.exp(-(distance_to_hill ** 2) / (hill_radius ** 2))
        terrain[hill_mask] += hill_height * hill_factor[hill_mask]
 
    # 3. 添加随机丘陵
    num_random_hills = 3
    for _ in range(num_random_hills):
        attempts = 0
        while attempts < 10:
            center_x = np.random.uniform(trench_start_x, trench_end_x)
            center_y = np.random.uniform(-size_y * 0.6, size_y * 0.6)

            # 检查是否远离沟壑
            distance_to_trench = np.min(np.abs(center_y - amplitude * np.sin(2 * np.pi * x / wavelength)))
            if distance_to_trench > trench_width * 1.2:
                break
            attempts += 1

        if attempts < 10:
            radius = np.random.uniform(1.5, 2.0)
            height = np.random.uniform(0.8, 1.6)
            distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
            mask = (distance < radius) & (~trench_region)
            factor = np.exp(-(distance ** 2) / (radius ** 2))
            terrain[mask] += height * factor[mask]

    # 4. 添加一些小山丘增加细节
    num_small_hills = 6
    for _ in range(num_small_hills):
        center_x = np.random.uniform(trench_start_x, trench_end_x)
        center_y = np.random.uniform(-size_y * 0.8, size_y * 0.8)
        radius = np.random.uniform(0.5, 1.5)
        height = np.random.uniform(0.1, 0.3)

        distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        mask = distance < radius
        factor = np.exp(-(distance ** 2) / (radius ** 2))
        terrain[mask] += height * factor[mask]

    # 5. 轻量平滑处理，保留更多细节
    terrain = ndimage.gaussian_filter(terrain, sigma=1.0)
 
    # 6. 添加小尺度噪声模拟自然地形（只在沟壑范围内）
    noise = np.random.rand(nrow, ncol) * 0.05
    trench_range_mask = (trench_start_x <= X) & (X <= trench_end_x)
    terrain[trench_range_mask] += noise[trench_range_mask]

    # 7. 最终轻量平滑
    terrain = ndimage.gaussian_filter(terrain, sigma=0.5)

    # 确保没有负高度
    terrain = np.maximum(terrain, 0)

    return terrain.ravel(), trench_params


def generate_sine_trench():
    """生成丰富的地形环境"""
    return generate_rich_environment()


def generate_complex_trench_system():
    """生成丰富的地形环境"""
    return generate_rich_environment()


def save_terrain_to_file(filename="rich_environment.txt"):
    """将地形数据保存到文件"""
    terrain_data, trench_params = generate_rich_environment()
    # 将 1D 数组 reshape 回 2D 以便更好地保存和查看
    terrain_2d = terrain_data.reshape(257, 257)
    np.savetxt(filename, terrain_2d)
    print(f"丰富地形环境数据已保存到 {filename}")
    return terrain_data, trench_params


def visualize_terrain(terrain_data, nrow, ncol, size_x, size_y):
    """
    使用 MuJoCo Viewer 独立可视化生成的地形
    无需依赖外部 XML 文件
    """
    # 构建一个极简的 MuJoCo XML 字符串来加载 heightfield
    xml_content = f"""
    <mujoco>
        <asset>
            <hfield name="terrain" nrow="{nrow}" ncol="{ncol}" size="{size_x} {size_y} 0.1 0.1"/>
        </asset>
        <worldbody>
            <light pos="0 0 15" dir="0 0 -1" directional="true"/>
            <geom type="hfield" hfield="terrain" pos="0 0 0" rgba="0.6 0.5 0.4 1"/>
        </worldbody>
    </mujoco>
    """
    
    print("正在启动 MuJoCo 可视化窗口... (按 ESC 或关闭窗口退出)")
    model = mujoco.MjModel.from_xml_string(xml_content)
    
    # 将生成的地形数据注入到模型中
    model.hfield_data[:] = terrain_data
    
    data = mujoco.MjData(model)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置一个较好的初始观察视角
        viewer.cam.lookat = [0, 0, 0]
        viewer.cam.distance = 20
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 90
        
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    print("开始生成地形...")
    
    # 1. 生成地形
    terrain_data, trench_params = generate_rich_environment()
    
    # 2. 打印参数
    print("\n--- 地形生成完毕 ---")
    print(f"沟壑起点 X: {trench_params['trench_start_x']}")
    print(f"沟壑终点 X: {trench_params['trench_end_x']}")
    print(f"幅度 (Amplitude): {trench_params['amplitude']:.2f}")
    print(f"波长 (Wavelength): {trench_params['wavelength']:.2f}")
    
    # 3. 保存到文件 (可选)
    save_terrain_to_file("rich_environment.txt")
    
    # 4. 可视化地形 (nrow=257, ncol=257, size_x=12.0, size_y=12.0)
    visualize_terrain(terrain_data, nrow=257, ncol=257, size_x=12.0, size_y=12.0)