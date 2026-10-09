"""Step 5: turn PyTorch weights into the two files the app loads: model.int8.onnx and tokens.txt.

    export_onnx.py <SenseVoiceSmall dir> <weights model.pt or -> <out dir> [mix weight]

"-" exports the original model, which is how this script is checked: what comes out must
transcribe like the files the app downloads (README.md has the comparison). With a mix weight the weights are first
pulled back towards the original (mix.py).

The graph, its metadata and the quantization (dynamic, MatMul only, unsigned 8-bit weights) are
those of sherpa-onnx's own export, scripts/sense-voice/export-onnx.py in
https://github.com/k2-fsa/sherpa-onnx (Apache-2.0), from which this file is adapted; the phone's
runtime reads the front-end settings and the token ids of the prompt from that metadata.
"""
import os, sys
import sv


def main():
    model_dir, weights, out = sys.argv[1], sys.argv[2], sys.argv[3]
    mix = float(sys.argv[4]) if len(sys.argv) > 4 else None

    import onnx, torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    os.makedirs(out, exist_ok=True)
    m = sv.SenseVoice(model_dir, weights=None if weights == "-" else weights, mix=mix)
    model = m.model

    class Graph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed, self.encoder, self.ctc_lo = model.embed, model.encoder, model.ctc.ctc_lo

        def forward(self, x, x_length, language, text_norm):
            language_query = self.embed(language).unsqueeze(1)
            text_norm_query = self.embed(text_norm).unsqueeze(1)
            event_emo_query = self.embed(torch.LongTensor([[1, 2]])).repeat(x.size(0), 1, 1)
            x = torch.cat((language_query, event_emo_query, text_norm_query, x), dim=1)
            encoder_out = self.encoder(x, x_length + 4)[0]
            return self.ctc_lo(encoder_out)

    with open(os.path.join(out, "tokens.txt"), "w", encoding="utf-8") as f:
        for i in range(m.sp.vocab_size()):
            f.write(f"{m.sp.id_to_piece(i)} {i}\n")

    torch.manual_seed(20240717)
    full = os.path.join(out, "model.onnx")
    with torch.no_grad():
        torch.onnx.export(
            Graph().eval(),
            (torch.randn(2, 100, 560), torch.tensor([80, 100], dtype=torch.int32),
             torch.tensor([0, 3], dtype=torch.int32), torch.tensor([14, 15], dtype=torch.int32)),
            full,
            opset_version=13,
            input_names=["x", "x_length", "language", "text_norm"],
            output_names=["logits"],
            dynamic_axes={"x": {0: "N", 1: "T"}, "x_length": {0: "N"}, "language": {0: "N"},
                          "text_norm": {0: "N"}, "logits": {0: "N", 1: "T"}},
            dynamo=False,
        )

    # the numbers as am.mvn spells them, like sherpa-onnx's export
    with open(os.path.join(model_dir, "am.mvn"), encoding="utf-8") as f:
        neg_mean, inv_stddev = [",".join(line.split()[3:-1]) for line in f if line.startswith("<LearnRateCoef>")]
    meta = {
        "lfr_window_size": sv.LFR_WINDOW,
        "lfr_window_shift": sv.LFR_SHIFT,
        "normalize_samples": 0,  # samples are expected in [-32768, 32767]
        "neg_mean": neg_mean,
        "inv_stddev": inv_stddev,
        "model_type": "sense_voice_ctc",
        "version": "2",  # unsigned 8-bit weights
        "model_author": "iic",
        "maintainer": "local-voice-ime",
        "vocab_size": m.sp.vocab_size(),
        "comment": "iic/SenseVoiceSmall, fine-tuned",
        "lang_auto": model.lid_dict["auto"],
        "lang_zh": model.lid_dict["zh"],
        "lang_en": model.lid_dict["en"],
        "lang_yue": model.lid_dict["yue"],
        "lang_ja": model.lid_dict["ja"],
        "lang_ko": model.lid_dict["ko"],
        "lang_nospeech": model.lid_dict["nospeech"],
        "with_itn": model.textnorm_dict["withitn"],
        "without_itn": model.textnorm_dict["woitn"],
        "url": "https://huggingface.co/FunAudioLLM/SenseVoiceSmall",
    }
    graph = onnx.load(full)
    del graph.metadata_props[:]
    for key, value in meta.items():
        prop = graph.metadata_props.add()
        prop.key, prop.value = key, str(value)
    onnx.save(graph, full)

    quantize_dynamic(model_input=full, model_output=os.path.join(out, "model.int8.onnx"),
                     op_types_to_quantize=["MatMul"], weight_type=QuantType.QUInt8)
    for name in ("model.onnx", "model.int8.onnx", "tokens.txt"):
        print(f"{os.path.getsize(os.path.join(out, name)):>11d}  {os.path.join(out, name)}")


if __name__ == "__main__":
    main()
