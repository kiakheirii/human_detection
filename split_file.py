import splitfolders

splitfolders.ratio(
    input=r"C:\Users\noavar\Desktop\uni_project\labeled_images",
    output="output_dataset",
    seed=42,
    ratio=(0.7, 0.15, 0.15)
)
