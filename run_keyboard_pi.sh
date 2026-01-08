ROOT_DIR="/data_248/pdss/hospital_data"
SCRIPT="./extract_keyboard_pi.py" # script path
HOLDTIME_MODE="paired_rows"       

python "$SCRIPT" \
  --root_dir "$ROOT_DIR" \
  --holdtime_mode "$HOLDTIME_MODE"
