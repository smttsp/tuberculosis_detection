import sys

from model_cnn.evaluation import get_model_prediction


if __name__ == "__main__":
    image_path = sys.argv[1]
    print(image_path)
    # image_path = "/Users/samet/Documents/data/TB_Chest_Radiography_Database/Normal/Normal-1821.png"
    get_model_prediction(image_path)
