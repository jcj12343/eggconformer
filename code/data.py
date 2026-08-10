import os
import mne
import numpy as np

# ===================== 全局目录配置（统一根路径） =====================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
RAW_DATA_ROOT = r"C:\Users\jcj\Downloads\BCICIV_2a_gdf"
SAVE_PROCESSED_ROOT = os.path.join(ROOT, "data", "processed")

SUBJECT_LIST = [f"A{i:02d}" for i in range(1, 10)]
SESSION_TYPE = ["T", "E"]
FS = 250
HP_FREQ = 4
LP_FREQ = 30
NOTCH_FREQ = 50
# Epoch窗口（必须包含基线区间）
TMIN_EPOCH = -0.5
TMAX_EPOCH = 2.5
BASELINE_WINDOW = (-0.5, 0.0)
# 送入模型的有效区间
CROP_TMIN = 0.0
CROP_TMAX = 2.5
REJECT_THRESH = 60e-6
# E session固定标签序列，总共288个trial
E_LABEL_SEQUENCE = np.tile([0, 1, 2, 3], reps=72)
assert len(E_LABEL_SEQUENCE) == 288
# T session事件映射
T_EVENT_DICT = {
    '769': 0,  # Left
    '770': 1,  # Right
    '771': 2,  # Feet
    '772': 3   # Tongue
}

# 子文件夹
fif_dir = os.path.join(SAVE_PROCESSED_ROOT, "epo_fif")
npy_dir = os.path.join(SAVE_PROCESSED_ROOT, "norm_data")
os.makedirs(fif_dir, exist_ok=True)
os.makedirs(npy_dir, exist_ok=True)

def preprocess_single_gdf(gdf_path, sub_id, session):
    print(f"\n==== Processing {sub_id}{session}.gdf ====")
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    # 修复重复通道名
    ch_names = raw.ch_names
    new_ch_names = []
    cnt = 1
    for name in ch_names:
        if name == "EEG":
            new_ch_names.append(f"EEG{cnt:02d}")
            cnt += 1
        else:
            new_ch_names.append(name)
    raw.rename_channels(dict(zip(ch_names, new_ch_names)))
    # 滤波与重参考
    raw.set_eeg_reference("average", verbose=False)
    raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)
    raw.filter(l_freq=HP_FREQ, h_freq=LP_FREQ, verbose=False)
    if session == "T":
        # T集：正常读取事件标记
        annot = raw.annotations
        onset_times = annot.onset
        desc_texts = annot.description
        ev_list = []
        for t, desc in zip(onset_times, desc_texts):
            if desc in T_EVENT_DICT:
                sample_idx = int(round(t * raw.info['sfreq']))
                ev_list.append([sample_idx, 0, T_EVENT_DICT[desc]])
        if len(ev_list) == 0:
            raise RuntimeError(f"{sub_id}{session} 未找到运动想象标记！")
        events = np.array(ev_list, dtype=int)
        labels = events[:, 2]
        epochs = mne.Epochs(
            raw,
            events=events,
            tmin=TMIN_EPOCH,
            tmax=TMAX_EPOCH,
            baseline=BASELINE_WINDOW,
            preload=True,
            verbose=False,
            reject={"eeg": REJECT_THRESH}
        )
    else:
        # E集：依靠768标记截取
        annot = raw.annotations
        onset_times = annot.onset
        desc_texts = annot.description
        ev_list = []
        # 筛选全部768触发点
        for t, desc in zip(onset_times, desc_texts):
            if desc == "768":
                sample_idx = int(round(t * raw.info['sfreq']))
                ev_list.append([sample_idx, 0, 99])
        events = np.array(ev_list, dtype=int)
        print(f"E集识别到768标记总数：{len(events)}")
        if len(events) != 288:
            print(f"⚠️警告：{sub_id}E.gdf 768数量不等于288！实际={len(events)}")
        # 按顺序绑定固定标签序列
        labels = E_LABEL_SEQUENCE[:len(events)]
        epochs = mne.Epochs(
            raw,
            events=events,
            tmin=TMIN_EPOCH,
            tmax=TMAX_EPOCH,
            baseline=BASELINE_WINDOW,
            preload=True,
            verbose=False,
            reject={"eeg": REJECT_THRESH}
        )
    # 基线校正后裁剪有效区间
    epochs.crop(tmin=CROP_TMIN, tmax=CROP_TMAX)
    data = epochs.get_data()[:, :22, :]
    clean_labels = labels[epochs.selection]
    print(f"有效Epoch数量：{len(clean_labels)} | data shape: {data.shape}")
    # 归一化，防止数据泄露
    if session == "T":
        mean = np.mean(data, axis=(0, 2), keepdims=True)
        std = np.std(data, axis=(0, 2), keepdims=True)
        np.save(os.path.join(npy_dir, f"{sub_id}_mean.npy"), mean)
        np.save(os.path.join(npy_dir, f"{sub_id}_std.npy"), std)
        data_norm = (data - mean) / std
    else:
        mean = np.load(os.path.join(npy_dir, f"{sub_id}_mean.npy"))
        std = np.load(os.path.join(npy_dir, f"{sub_id}_std.npy"))
        data_norm = (data - mean) / std
    epochs.save(os.path.join(fif_dir, f"{sub_id}{session}_epo.fif"), overwrite=True)
    np.save(os.path.join(npy_dir, f"{sub_id}{session}_data.npy"), data_norm)
    np.save(os.path.join(npy_dir, f"{sub_id}{session}_label.npy"), clean_labels)
    return

if __name__ == "__main__":
    # 自动创建全部目录
    os.makedirs(RAW_DATA_ROOT, exist_ok=True)
    os.makedirs(fif_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    for sub in SUBJECT_LIST:
        for sess in SESSION_TYPE:
            filepath = os.path.join(RAW_DATA_ROOT, f"{sub}{sess}.gdf")
            if os.path.exists(filepath):
                try:
                    preprocess_single_gdf(filepath, sub, sess)
                except Exception as e:
                    print(f"处理 {sub}{sess} 失败：{str(e)}\n")
            else:
                print(f"缺失文件：{filepath}")
    print("====全部预处理流程结束====")