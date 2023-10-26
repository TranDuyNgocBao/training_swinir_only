# Clone repo
```
Bước 1: git clone https://github.com/TranDuyNgocBao/training_swinir_only.git
Bước 2: Di chuyển tới thư mục SwinIR
        cd SwinIR
```

# Requirements
Cài đặt thư viện
```
pip install -r requirements.txt
```

Bên trong thư mục `SwinIR`:
# Data
* Tải `data_sets.zip` tại link: [GG drive](https://drive.google.com/file/d/1VQJonF_wdOHQV-ZLzKVvfwT9GW5X_mgA/view?usp=sharing)
* Tiếp theo giải nén data vào thư mục `datasets`
* Cấu trúc của thư mục `datasets` sẽ bao gồm:
    ```
    datasets
    ├───combine_data_training   // data training
    └───hr_val_2                     // data val (5 ảnh từ tập val USR248)
    └───Lr_val_2                     // data val (5 ảnh từ tập val USR248)
    ```

# Pretrain model
* Tải `pretrain.zip` tại link: [GG drive](https://drive.google.com/file/d/1BCGK1KDacNaz_AATHEEXUMq729WJYfu9/view?usp=sharing)
* Tiếp theo giải nén file pretrain bỏ vào thư mục `model_zoo`
* Cấu trúc của thư mục `model_zoo` sẽ bao gồm:
    ```
    model_zoo
    ├───pretrain
    └───README(1).md
    ```
# Training

* Train SwinIR với multiprocessing (tuy nhiên thường gặp lỗi conflict giữa các thiết bị)
```
torchrun --standalone --nnodes=1 --nproc_per_node=1 main_train_psnr.py --opt swinir_training/psnr_train_swinir_sr_realworld_x4_default.json  --dist True
```
* Train SwinIR với đơn processing (ổn định hơn, ít gặp lỗi hơn)
```
python main_train_psnr.py --opt swinir_training/psnr_train_swinir_sr_realworld_x4_default.json  --dist True
```

**Note:** Cần chờ khoảng 1 phút từ lúc bắt đàu train để xác định train có được hay không? Sau một phút thì mô hình tự học không gặp lỗi \\
**Note:** Các trọng số được lưu tại thư mục `model_zoo`
