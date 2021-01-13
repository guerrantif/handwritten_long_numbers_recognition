"""
Copyright 2021 - Filippo Guerranti <filippo.guerranti@student.unisi.it>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import modules.utils as utils


def main():
    args = utils.main_args_parser()

    if args.mode == "classify":
        utils.classify(args.folder, args.augmentation, args.model, args.device)

    elif args.mode == "train":
        utils.train(args.augmentation, args.splits, args.batch_size, args.epochs, args.learning_rate, args.num_workers, args.device)

    elif args.mode == "eval":
        utils.eval(args.model, args.device)

    else:
        raise ValueError("Execution mode must be either (classify) or (train)")




if __name__ == "__main__":
    main()


# models/CNN-batch_size64-lr0.001-epochs50-a.pth
# models/CNN-batch_size64-lr0.001-epochs50.pth
# models/CNN-batch_size128-lr0.001-epochs60-a.pth
# models/CNN-batch_size128-lr0.001-epochs60.pth
# models/CNN-batch_size150-lr0.001-epochs40-a.pth
# models/CNN-batch_size150-lr0.001-epochs40.pth