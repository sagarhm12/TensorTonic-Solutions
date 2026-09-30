class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:

        text_lower = [text.lower() for text in texts]

        vocab_set = set()

        for text in text_lower:
            vocab_set.update(text.split())

        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]

        self.word_to_id = {
            token: idx
            for idx, token in enumerate(special_tokens)
        }

        for idx, word in enumerate(
            sorted(vocab_set),
            start=len(special_tokens)
        ):
            self.word_to_id[word] = idx

        self.id_to_word = {
            idx: word
            for word, idx in self.word_to_id.items()
        }

        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> list[int]:
        text = text.lower()
        words = text.split()

        encoded = []

        for word in words:
            token_id = self.word_to_id.get(
                word,
                self.word_to_id[self.unk_token]
            )

            encoded.append(token_id)

        return encoded

    def decode(self, ids: list[int]) -> str:

        decoded = []

        for token_id in ids:
            word = self.id_to_word.get(
                token_id,
                self.unk_token
            )

            decoded.append(word)

        return " ".join(decoded)