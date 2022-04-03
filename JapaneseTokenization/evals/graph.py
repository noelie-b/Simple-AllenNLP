from matplotlib import pyplot as plt
from pathlib import Path
import sys
import json

def main(data_path: Path):
    data = []
    for f in data_path.iterdir():
        if f.is_file() and f.name.startswith("metrics_epoch"):
            epoch_data = json.loads(f.read_text())
            e = epoch_data['epoch']
            training_loss = epoch_data['training_loss']
            validation_loss = epoch_data['validation_loss']
            data.append((e, training_loss, validation_loss))
        x, y1, y2 = zip(*data)
        plt.plot(x, y1, 'r.', x, y2, 'b.')
    #plt.title(f"itération {epoch}, exactitude:{score}")
    plt.title(f"itération : 50, LSTMSeq2Seq")
    # plt.savefig(f"./Plots/{name}{epoch:03}.png")
    plt.show()
    plt.close()


if __name__ == '__main__':
    main(Path("/tmp/test_classif2"))
