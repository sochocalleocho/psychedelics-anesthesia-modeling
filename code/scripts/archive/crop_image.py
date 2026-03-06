
from PIL import Image
import os

source_path = '/Users/soichi/.gemini/antigravity/brain/02630dfc-62ba-4789-871e-f2829c23a433/uploaded_media_1769788941554.png'
dest_path = '/Users/soichi/Desktop/Psychedelics & Anesthesia Modeling Study/mean_field_paper_plot.png'

if not os.path.exists(source_path):
    print(f"Error: {source_path} not found.")
    exit(1)

img = Image.open(source_path)
width, height = img.size
print(f"Original size: {width}x{height}")

# Estimated Grid
# 4 Columns of plots. 3 Rows.
# Margins estimated visually.
margin_left = 600  # The first two columns are Spiking Network (Raster + FR)
col_width = 440    # Width of the Mean Field column
row_height = 420   # Approx height of one row
margin_top = 150   # Header
row_gap = 50

# We want Row 3 (index 2) -> "c"
# We want Column 3 (index 0 relative to where we start counting?) 
# Labels: Spiking(2 cols) | MeanField(1 col) | WholeBrain(1 col)
# So Mean Field is the 3rd column overall.

# X Start: margin_left is approx where Col 3 starts?
# Let's count pixels.
# Total width 1800. 4 cols -> 450 per col.
# Col 1: 150-600
# Col 2: 600-1050
# Col 3: 1050-1500 (Mean Field)
# Col 4: 1500-1826 (Whole Brain)

x1 = 1020
x2 = 1430

# Y Start
# Row 1 (a): 150 - 550
# Row 2 (b): 550 - 950
# Row 3 (c): 950 - 1350

y1 = 1100
y2 = 1480

crop_box = (x1, y1, x2, y2)
cropped = img.crop(crop_box)
cropped.save(dest_path)

print(f"Cropped saved to {dest_path}")
print(f"Crop Stats: Mean Color {Image.eval(cropped, lambda x: x).getextrema()}")
