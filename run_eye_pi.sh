python /data_248/pdss/primitive_indicator_scripts/extract_eye-tracking_pi.py \
  --base_path /data_248/pdss/hospital_data/ \
  --output_base_path /data_248/pdss/primitive_indicator \
  #--clients 1d8da2cc-b4dd-4841-94f9-4deb44527526 \
  --pi_excel /data_248/pdss/primitive_indicator_scripts/Primitive_Indicator_eye.xlsx \
  --pi_sheet eye-tracking
  --window_size 60 \
  --stride 30 \
  --screen_width 1440 \
  --screen_height 900