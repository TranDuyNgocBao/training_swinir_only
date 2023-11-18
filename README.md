# Clone repo
```
Bước 1: git clone https://github.com/TranDuyNgocBao/training_swinir_only.git
Bước 2: Di chuyển tới thư mục SwinIR
        cd training_swinir_only/SwinIR
```

Bên trong thư mục `training_swinir_only/SwinIR`:

# Requirements
Cài đặt thư viện
```
pip install -r requirements.txt
```

# Data
* Tải `data_sets.zip` tại link: [GG drive](https://drive.google.com/file/d/1VQJonF_wdOHQV-ZLzKVvfwT9GW5X_mgA/view?usp=sharing)
* Tiếp theo giải nén data sẽ có thư mục `datasets` bên trong và cho vào thư mục `SwinIR/datasets`
* Chỉ cần bỏ đúng thư mục `datasets` vào `SwinIR` thì sẽ tương tự đường dẫn trong file json.
* Cấu trúc của thư mục `datasets` sẽ bao gồm:
    ```
    datasets
    ├───combine_data_training   // data training
    └───hr_val_2                     // data val (5 ảnh từ tập val USR248)
    └───lr_val_2                     // data val (5 ảnh từ tập val USR248)
    ```

# Pretrain model
* Tải folder `swinir_sr_realworld_x4_psnr` tại link: [GG drive](https://drive.google.com/drive/folders/1OzSKCk57-NLC4-03EoVxBDg7jf8xM5AP?usp=sharing)
* Đặt folder vừa tải theo path `SwinIR/model_zoo/swinir_sr_realworld_x4_psnr` 
* Cấu trúc của thư mục `model_zoo` sẽ bao gồm:
    ```
    model_zoo
    ├───swinir_sr_realworld_x4_psnr
    └───README(1).md
    ```
# Training
- Có 3 files json trong `swinir_training`, chúng ta sẽ train 2 files, trong câu lệnh training bên dưới thì lần lượt đổi đường dẫn của `--opt` thành: `swinir_training/psnr_train_swinir_sr_realworld_x4_large.json` và `swinir_training/psnr_train_swinir_sr_realworld_x4_beginner.json`

- Chỉ chọn một trong 3 lệnh bên dưới để training (1 trong 3 options)
  
* Train SwinIR với multiprocessing (tuy nhiên thường gặp lỗi conflict giữa các thiết bị)
```
torchrun --standalone --nnodes=1 --nproc_per_node=8 main_train_psnr_2.py --opt swinir_training/psnr_train_swinir_sr_realworld_x4_default.json
# thay đổi nproc_per_node=8 để lựa chọn số processes mỗi unit.
```
hoặc
```
torchrun --standalone --nnodes=1 --nproc_per_node=1 main_train_psnr_2.py --opt swinir_training/psnr_train_swinir_sr_realworld_x4_default.json
# như bên dưới
```
* Train SwinIR với đơn processing (ổn định hơn, ít gặp lỗi hơn)
```
python main_train_psnr_2.py --opt swinir_training/psnr_train_swinir_sr_realworld_x4_default.json
```

**Note:** Cần chờ khoảng 1 phút từ lúc bắt đàu train để xác định train có được hay không? Sau một phút thì mô hình tự học không gặp lỗi **
**Note:** Các trọng số được lưu tại thư mục `model_zoo`
