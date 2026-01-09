<!-- Template source: See: https://github.com/othneildrew/Best-README-Template -->
<a id="readme-top"></a>

[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]](https://www.linkedin.com/in/arthur-testard/)


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- <a href="https://github.com/art-test-stack/gpt-lib">
    <img src="rsc/logo.jpg" alt="Logo" height="350">
  </a> -->

<h3 align="center">Generative Pre-trained Transformer Library</h3>

  <p align="center">
    This project is the implementation of a light-weight library for LLM management and monitoring, from training to inference. It also includes an interface to chat with the model, and with models from 🤗 API, locally or remotly.
    <br />
    <a href="https://github.com/art-test-stack/gpt-lib"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/art-test-stack/gpt-lib/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- ### The implementation -->


### Built With

* [![Torch][Torch]][Torch-url] <<3
* [![huggingface-shield]][huggingface-url] (datasets, transformers, tokenizer, hub)
* [![gradio-shield]][gradio-url] (web interface)
* [![tiktoken-shield]][tiktoken-url] (tokenizer)

### Roadmap 

* Tokenization 
  - BPE implementation in Python
  - Rust implementation
* Positional embedding
  - Absolute
  - rotary
* Transformer
  - Attention mechanism
  - Multihead attention
  - flash attention
  - FFN, RMSNorm layers
* Training
  - Pre-training
  - fine-tuning
  - intruction tuning
  - rlhf, dpo
  - ddp, fsdp method
* Sampling
  - temperature
  - top-k, top-p
  - beam-search
* Too move beyond
  - KV-cache
  - sliding window
  - memory layers?
  - MoE
  - Quantization
* Training on Synthetic Data
  - generate data
  - model teacher

## Get Started

This project has been developed and tested with Python 3.12. To manage dependencies, I recommend using [`uv`](https://github.com/astral-sh/uv). 

1. Clone the repo
   ```sh
   git clone git@github.com:art-test-stack/gpt-lib.git
   ```
2. Install dependencies
   ```sh
    uv sync
   ```
   If running on Linux with CUDA available, you can install the GPU version of PyTorch by running:
    ```sh
    uv sync --extra cuda
     ```

> [!NOTE]  
> Make sure to adjust the CUDA version in `uv.toml` if needed. This extra is only available for Linux systems with compatible NVIDIA GPUs. It permits using `flash_attention` for faster attention computation.


### Training a model

Coming soon...

### Chat with the model

In this section, you will find instructions to run the chat interface with different models.

Under development environment (`ENV='development'` in `.env`), you can run the chat interface with auto-reloading, use the following command:
```sh
uv run gradio scripts/chat_app.py --demo-name=app
```

Otherwise, if you don't want auto-reloading, use:
```sh
uv run python -m scripts.chat_app
```

Then, open your browser and go to [`http://127.0.0.1:7860/`](http://127.0.0.1:7860/). It is quite straightforward to use. You can select different models (local or remote), choose some hyperparameters for inference, and chat with the model.

## Data

### Pre-training Data Summary
<!-- <table>
    <thead>
        <tr>
            <th align="center">Source</th>
            <th align="center">Documents</th>
            <th align="center">Tokens</th>
            <th align="center">Ratio</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="left"><b><a href="https://wikipedia.org/">Wikipedia</a></b></td>
            <td align="center">-</td>
            <td align="center">-</td>
            <td align="center">-</td>
        </tr>
        <tr>
            <td align="left"><b><a href="https://huggingface.co/datasets/Skylion007/openwebtext">OpenWebText</a></b></td>
            <td align="center">-</td>
            <td align="center">-</td>
            <td align="center">-</td>
        </tr>
        <tr>
            <th align="left">Total</th>
            <th align="center">-</th>
            <th align="center">-</th>
            <th align="center">100.00 %</th>
        </tr>
    </tbody>
</table> -->

<!-- Sources -->
## Sources

1. [Attention is all you need](https://arxiv.org/pdf/1706.03762)
2. [Building a text generation model from scratch by Vincent Bons](https://wingedsheep.com/building-a-language-model/)
3. [nanoGPT by Andrej Karpathy](https://github.com/karpathy/nanoGPT/tree/master) 
4. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
5. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.



<!-- CONTACT -->
## Contact

Arthur Testard - [arthur.testard.pro@gmail.com](mailto:arthur.testard.pro@gmail.com)

Project Link: [https://github.com/art-test-stack/gpt-lib](https://github.com/art-test-stack/gpt-lib)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/art-test-stack/gpt-lib.svg?style=for-the-badge
[contributors-url]: https://github.com/art-test-stack/gpt-lib/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/art-test-stack/gpt-lib.svg?style=for-the-badge
[forks-url]: https://github.com/art-test-stack/gpt-lib/network/members
[stars-shield]: https://img.shields.io/github/stars/art-test-stack/gpt-lib.svg?style=for-the-badge
[stars-url]: https://github.com/art-test-stack/gpt-lib/stargazers
[issues-shield]: https://img.shields.io/github/issues/art-test-stack/gpt-lib.svg?style=for-the-badge
[issues-url]: https://github.com/art-test-stack/gpt-lib/issues
[license-shield]: https://img.shields.io/github/license/art-test-stack/gpt-lib.svg?style=for-the-badge
[license-url]: https://github.com/art-test-stack/gpt-lib/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/arthur-testard
[product-screenshot]: images/screenshot.png
[Torch]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[Torch-url]: https://pytorch.org/
[huggingface-shield]: https://img.shields.io/badge/HuggingFace-%23FF6C37.svg?style=for-the-badge&logo=HuggingFace&logoColor=white
[huggingface-url]: https://huggingface.co/
[gradio-shield]: https://img.shields.io/badge/Gradio-%23FF6C37.svg?style=for-the-badge&logo=Gradio&logoColor=white
[gradio-url]: https://gradio.app/
[tiktoken-shield]: https://img.shields.io/badge/tiktoken-%23007ACC.svg?style=for-the-badge&logo=tiktoken&logoColor=white
[tiktoken-url]: https://github.com/openai/tiktoken