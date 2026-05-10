import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from transformers import CLIPTextModel
from diffusers import StableDiffusionPipeline


class AddSpecialToken:
    """
    A class to add a special token to a tokeninzer and intialize its emdedding
    It takes StableDiffusionPipeline object as inputs andd add a new token to the tokenizer.
    It also initializes the new token's embedding from a set of reference words.
    """
    def __init__(self, pipeline: StableDiffusionPipeline, style_token: str):
        self.pipeline: StableDiffusionPipeline = pipeline
        self.style_token: str = style_token
        self.tokenizer: PreTrainedTokenizerBase = pipeline.tokenizer
        self.text_encoder: CLIPTextModel = pipeline.text_encoder
    
    @torch.no_grad()
    def apply(self, ref_words=("anime", "illustration", "ghibli")):
        """Apply the special token addition and embedding initialization."""
        self.new_token_id: int = self.add_style_token(self.style_token)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.init_token_embedding_from_words(ref_words=ref_words)
        
    def add_style_token(self, token: str) -> int:
        """Add `token` to tokenizer if missing. Returns its token id."""
        # If already present, just return its ID
        if token in self.tokenizer.get_vocab():
            return self.tokenizer.convert_tokens_to_ids(token)

        # Add as an 'additional_special_token' so it won't be split
        num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": [token]})
        if num_added == 0:  # tokenizer may report 0 if it was already there
            return self.tokenizer.convert_tokens_to_ids(token)

        return self.tokenizer.convert_tokens_to_ids(token)
    
    def init_token_embedding_from_words(self,
                                        ref_words=("anime", "illustration", "ghibli")
                                        ):
        # Grab the input embedding table
        emb = self.text_encoder.get_input_embeddings()   # nn.Embedding
        new_id = self.tokenizer.convert_tokens_to_ids(self.style_token)
        device = emb.weight.device

        vecs = []
        for w in ref_words:
            # Tokenize (may produce multiple ids), get their embeddings and average
            ids = self.tokenizer(w, add_special_tokens=False)["input_ids"]
            if len(ids) == 0:
                continue
            idx = torch.tensor(ids, device=device, dtype=torch.long)
            w_vec = emb.weight.index_select(0, idx).mean(dim=0)  # device-safe
            vecs.append(w_vec)

        if len(vecs) == 0:
            print(f"[warn] No valid reference tokens found; leaving {self.style_token} random-initialized.")
            return

        init_vec = torch.stack(vecs, dim=0).mean(dim=0)
        # (Optional) normalize a bit to typical embedding scale
        init_vec = F.normalize(init_vec, dim=0) * emb.weight.data.std()

        # safe in-place write under no_grad
        emb.weight[new_id].copy_(init_vec)
        print(f"Initialized embedding for {self.style_token} from refs: {ref_words}")

if __name__ == "__main__":

    MODEL_NAME     = "runwayml/stable-diffusion-v1-5"
    style_token = "<sks>"   # your new style token
    device         = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    AddSpecialToken(pipe, style_token).apply()

    # tokenizer     = pipe.tokenizer          # CLIPTokenizer (from transformers)
    # text_encoder  = pipe.text_encoder       # CLIPTextModel

