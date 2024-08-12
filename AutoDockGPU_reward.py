import os
import subprocess
import shutil
import tempfile

from meeko import MoleculePreparation
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Geometry import Point3D

from chemtsv2.reward import Reward

class AutoDockGPU_reward(Reward):
    """
    计算 AutoDock-GPU 对接分数作为奖励值。
    """

    def get_objective_functions(conf):
        """
        定义目标函数。
        """
        def AutoDockGPUScore(mol):
            """
            使用 AutoDock-GPU 计算单个分子的对接分数。
            """
            verbosity = 1 if conf['debug'] else 0
            temp_dir = tempfile.mkdtemp()
            temp_ligand_fname = os.path.join(temp_dir, 'ligand_temp.pdbqt')
            pose_dir = os.path.join(conf['output_dir'], "3D_pose")
            os.makedirs(pose_dir, exist_ok=True)
            # 修改 output_dlg_fname，去掉 .dlg 后缀
            output_dlg_fname = os.path.join(pose_dir, f"mol_{conf['gid']}_out")  

            # 使用 RDKit 处理分子
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            try:
                mol_conf = mol.GetConformer(-1)
            except ValueError as e:
                print(f"Error SMILES: {Chem.MolToSmiles(mol)}")
                print(e)
                if not conf['debug']:
                    shutil.rmtree(temp_dir)
                return None

            # 使用 meeko 处理分子并保存为 PDBQT 文件
            mol_prep = MoleculePreparation()
            mol_prep.prepare(mol)
            mol_prep.write_pdbqt_file(temp_ligand_fname)

            # 构建 AutoDock-GPU 命令行调用语句
            cmd = [
                conf['autodock_gpu_bin_path'],  # AutoDock-GPU 可执行文件路径
                '--ffile', conf['autodock_gpu_receptor'],  # 受体文件路径
                '--lfile', temp_ligand_fname,  # 配体文件路径
                '--nrun', str(conf['autodock_gpu_nruns']),  # 运行次数
                '--resnam', output_dlg_fname,  # 指定输出文件名（不带 .dlg 后缀）
                '--derivtype', 'S=SA',  # 将硫原子衍生为碳原子，注意添加逗号
                # ... (其他 AutoDock-GPU 参数)
            ]

            if conf['debug']:
                print(cmd)

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error SMILES: {Chem.MolToSmiles(mol)}")
                print(e)
                if not conf['debug']:
                    shutil.rmtree(temp_dir)
                return None

            # 读取 AutoDock-GPU 输出文件，提取对接分数
            # 注意：output_dlg_fname 现在不包含 .dlg 后缀，需要手动添加
            with open(output_dlg_fname + ".dlg", 'r') as f:
                for line in f:
                    if 'RMSD TABLE' in line:
                        score_line = f.readlines()[9]
                        score = float(score_line.split()[3])
                        break

            # 删除临时目录
            if not conf['debug']:
                shutil.rmtree(temp_dir)

            return score  # 返回对接分数

        return [AutoDockGPUScore]

    def calc_reward_from_objective_values(values, conf):
        """
        根据 AutoDock-GPU 分数计算奖励值。
        """
        min_inter_score = values[0]
        if min_inter_score is None:
            return -1

       # 参考 Vina 的奖励计算方法
        score_diff = min_inter_score - conf['autodock_gpu_base_score']  # 使用 autodock_gpu_base_score
        reward = - score_diff * 0.1 / (1 + abs(score_diff) * 0.1)

        return reward  # 返回奖励值