class CharacterTokenizer:
    def __init__(self, text: str):
        # Extract all unique characters
        characters = sorted(set(text))

        # Character -> integer
        self.stoi = {char: index for index, char in enumerate(characters)}

        # Integer -> character
        self.itos = {index: char for char, index in self.stoi.items()}

        self.vocab_size = len(characters)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos[token] for token in tokens)


if __name__ == "__main__":
    training_text = "hello mike, welcome to language models!"

    tokenizer = CharacterTokenizer(training_text)

    encoded = tokenizer.encode("hello mike")
    decoded = tokenizer.decode(encoded)

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    assert decoded == "hello mike"

    # print("\nExperiment: Unknown characters")
    # encoded = tokenizer.encode("hello python")
    # print(encoded)
