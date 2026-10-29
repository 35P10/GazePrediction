"""UMSS stage 1: fine-tune MD-SEM (CodeCharts weights) on MASSVIS-EFD 500/3000/5000 ms.

Executes the original training cells of `UMSS-S1.ipynb` (DaRUS V2) through
models/umss/training/s1.py. Model settings, generators, losses, optimizer, callbacks
and fitting calls come from the original notebook. Runtime adaptations:
  * 6 epochs, as stated in the paper (Sec. 5.2); the public notebook says 9 (`--epochs 9`);
  * the notebook's InteractivePlot callback (live matplotlib plot) is replaced by a CSV log;
  * the generator class is imported explicitly (the notebook relies on a missing import);
  * optional seeds (the notebook sets none);
  * after training, the weights of the last epoch are saved as md_eam.hdf5: the notebook predicts
    with the model left in memory by fit_generator (all epochs are also saved, as upstream does);
  * `--micro-batch m` (< 8) for GPUs without room for 8 images: the same batch-8 Adam updates
    built from 8/m accumulated micro-batches (see accumulation.py; BatchNorm statistics differ).
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

def main(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    code = os.path.join(args.work, 'upstream', 'Code', 'scripts')
    sys.path[:0] = [os.path.join(code, 'src'), os.path.join(code, 'saliency'),
                    os.path.join(args.work, 'downloads', 'extra_src')]
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

    repository = Path(args.work) / 'upstream'

    import numpy as np
    import tensorflow as tf
    import keras
    from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
    from keras.optimizers import Adam
    from data_loading import load_datasets_multiduration  # official
    from util import get_model_by_name, create_losses  # official
    from sal_imp_utilities import MultidurationGenerator  # official
    from training.s1 import run_s1, s1_defaults

    recipe = s1_defaults(repository)
    print('TensorFlow', tf.__version__, '| Keras', keras.__version__, '| GPU:', tf.test.is_gpu_available())
    if args.micro_batch < recipe['batch_size']:
        # Runs first: it resets the Keras session, which would drop the graph seed below.
        from accumulation import adam_accumulate_class, accumulation_sequence, selftest
        selftest()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    ckpt_savedir = os.path.join(args.work, 'ckpt')
    bindings = dict(
        bp=os.path.join(args.work, 'datasets'), ckpt_savedir=ckpt_savedir,
        weightspath=os.path.join(args.work, 'downloads', 'mdsem_codecharts0_cameraready_weights.hdf5'),
        n_epochs=args.epochs, batch_size=args.micro_batch, training_verbose=args.verbose,
        Adam=Adam, ModelCheckpoint=ModelCheckpoint, LearningRateScheduler=LearningRateScheduler,
        InteractivePlot=lambda: CSVLogger(os.path.join(ckpt_savedir, 'training_log.csv')),
        load_datasets_multiduration=load_datasets_multiduration,
        get_model_by_name=get_model_by_name, create_losses=create_losses,
        MultidurationGenerator=MultidurationGenerator)
    wrap_generator = None
    if args.micro_batch < recipe['batch_size']:
        bindings['opt'] = adam_accumulate_class()(
            accum_steps=recipe['batch_size'] // args.micro_batch, lr=recipe['init_lr'])
        wrap_generator = lambda generator: accumulation_sequence(
            generator, recipe['batch_size'], args.micro_batch)
    state = run_s1(repository, bindings, wrap_generator)

    # The notebook predicts with the in-memory model after fit_generator: the last epoch.
    checkpoint = os.path.join(ckpt_savedir, 'md_eam.hdf5')
    state['model'].save_weights(checkpoint)
    epochs = sorted(f for f in os.listdir(ckpt_savedir) if f.startswith(state['dataset']) and '_ep' in f)
    with open(os.path.join(ckpt_savedir, 'md_eam.json'), 'w') as f:
        json.dump(dict(selected='last epoch', epoch=args.epochs, seed=args.seed, epochs=args.epochs,
                       micro_batch=args.micro_batch, all=epochs), f, indent=2)
    print('Saved the last epoch (%d) -> %s' % (args.epochs, checkpoint))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work', default='/work')
    parser.add_argument('--epochs', type=int, default=6, help='6 in the paper (Sec. 5.2), 9 in UMSS-S1.ipynb')
    parser.add_argument('--seed', type=int, default=0, help='use -1 for no seeding, as in the notebook')
    parser.add_argument('--micro-batch', type=int, default=8, choices=[2, 4, 8],
                        help='images per forward pass; below 8, gradients are accumulated to batch 8')
    parser.add_argument('--verbose', type=int, default=2)
    args = parser.parse_args()
    if args.seed < 0:
        args.seed = None
    main(args)
