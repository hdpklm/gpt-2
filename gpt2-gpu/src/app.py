#!/usr/bin/env python3

# import fire
import json
import os
import time
import numpy as np
import tensorflow as tf
import threading
import model
import sample
import encoder
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='.')


class xState:
    def __init__(self):
        self.hash = []
        self.run_hash = 0
        self.fix_hash = 0
        self.text = None
        self.result = None

    def predict(self, text):
        hash = round(time.time() * 1000000)
        self.hash.append(hash)

        while self.hash[0] != hash:
            time.sleep(0.01)
            continue

        self.text = text
        while self.fix_hash != hash:
            time.sleep(0.01)
            continue

        return self.result


def getState():
    return state


@app.before_first_request
def interact_model():
    global state
    state = xState()

    """
        Interactively run the model
            :model_name=124M : String, which model to use
            :seed=None : Integer seed for random number generators, fix seed to reproduce results
            :nsamples=1 : Number of samples to return total
            :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
            :length=None : Number of tokens in generated text, if None (default), is
                determined by model hyperparameters
            :temperature=1 : Float value controlling randomness in boltzmann
                distribution. Lower temperature results in less random completions. As the
                temperature approaches zero, the model will become deterministic and
                repetitive. Higher temperature results in more random completions.
            :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
                considered for each step (token), resulting in deterministic completions,
                while 40 means 40 words are considered at each step. 0 (default) is a
                special setting meaning no restrictions. 40 generally is a good value.
            :models_dir : path to parent folder containing model subfolders
                (i.e. contains the <model_name> folder)
        """

    models_dir = 'models'
    model_name = '1558M'  # 124M  355M  774M  1558M
    seed = None
    nsamples = 1
    batch_size = 1
    length = 72
    temperature = 0.9
    top_k = 0
    top_p = 1

    def interact_model_thread(model_name, seed, nsamples, batch_size, length, temperature, top_k, top_p, models_dir, getState):
        models_dir = os.path.expanduser(os.path.expandvars(models_dir))
        if batch_size is None:
            batch_size = 1
        assert nsamples % batch_size == 0

        enc = encoder.get_encoder(model_name, models_dir)
        hparams = model.default_hparams()
        with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        if length is None:
            length = hparams.n_ctx // 2
        elif length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

        with tf.Session(graph=tf.Graph()) as sess:
            context = tf.placeholder(tf.int32, [batch_size, None])
            np.random.seed(seed)
            tf.set_random_seed(seed)
            output = sample.sample_sequence(
                hparams=hparams, length=length,
                context=context,
                batch_size=batch_size,
                temperature=temperature, top_k=top_k, top_p=top_p
            )

            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
            saver.restore(sess, ckpt)

            while True:
                state = getState()
                while len(state.hash) == 0:
                    time.sleep(0.01)
                    continue

                print("\n\nstart predict: %s" % state.text)
                context_tokens = enc.encode(state.text)
                len_ctx = len(context_tokens)
                context_tokens_arr = [context_tokens for _ in range(batch_size)]
                feeds = {context: context_tokens_arr}
                print("ctx: %s" % context_tokens)
                print("len_ctx: %s" % len_ctx)
                print("ctx_arr: %s" % context_tokens_arr)
                print("feeds: %s" % feeds)

                out = sess.run(output, feed_dict=feeds)
                print("out: %s" % out)

                outs = out[:, len_ctx:]
                print("outs: %s" % outs)

                state.result = enc.decode(outs[0])
                print(state.result)
                print("=" * 40 + " SAMPLE " + "=" * 40)

                state.fix_hash = state.hash.pop(0)
                continue

    _thread = threading.Thread(target=interact_model_thread, args=(model_name, seed, nsamples, batch_size, length, temperature, top_k, top_p, models_dir, getState))
    _thread.start()


@app.route("/")
def html():
    return render_template("public/index.html")


@ app.route("/msg", methods=["GET", "POST"])
def msg():
    if request.method == "POST":
        if not state:
            return {"error": "State is not initialized!"}, 400

        raw_text = request.get_json().get('text')
        rext = state.predict(raw_text)
        return jsonify({'text': rext})
    else:
        return {"error": "Only POST method is allowed!"}, 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
    # fire.Fire(interact_model)
