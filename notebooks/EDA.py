from pathlib import Path
import pandas as pd

# ========================
#  تحميل البيانات
# ========================

# تحديد مسار الملف (افتراض أنه في مجلد data في المشروع)
data_path = Path(__file__).resolve().parent.parent / "data" / "imu_data.csv"

if not data_path.exists():
    raise FileNotFoundError(f"⚠️ الملف غير موجود في: {data_path}")

df = pd.read_csv(data_path)

# ========================
# 1. معلومات أساسية
# ========================
print("🔹 Data Info:")
print(df.info())

print("\n🔹 Missing Values:")
print(df.isnull().sum())

print("\n🔹 Descriptive Statistics:")
print(df.describe().T)

# ========================
# 2. أول وأخر 5 صفوف
# ========================
print("\n🔹 First 5 rows:")
print(df.head())

print("\n🔹 Last 5 rows:")
print(df.tail())

# ========================
# 3. توزيع القيم (بشكل رقمي)
# ========================
print("\n🔹 Accelerometer stats:")
print(df[['accel_x','accel_y','accel_z']].describe().T)

print("\n🔹 Gyroscope stats:")
print(df[['gyro_x','gyro_y','gyro_z']].describe().T)

print("\n🔹 Magnetometer stats:")
print(df[['mag_x','mag_y','mag_z']].describe().T)

print("\n🔹 Position stats:")
print(df[['pos_x','pos_y','pos_z']].describe().T)

print("\n🔹 Orientation stats (Roll, Pitch, Yaw):")
print(df[['roll','pitch','yaw']].describe().T)
