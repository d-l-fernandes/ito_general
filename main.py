from absl import app
import trainers


def main(argv):
    trainer = trainers.ItoGeneralTrainer()
    trainer.train()


if __name__ == "__main__":
    app.run(main)
