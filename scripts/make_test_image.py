from PIL import Image

Image.new("RGB", (32, 32), "blue").save("test_image.png")
print("wrote test_image.png")
