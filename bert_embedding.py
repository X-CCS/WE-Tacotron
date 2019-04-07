import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


def add_cls_sep(text):
    return "[CLS] " + text + " [SEP]"


def get_bert_embedding(text, model, tokenizer):
    text = add_cls_sep(text)
    # print(text)
    tokenized_text = tokenizer.tokenize(text)
    # print(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # print(indexed_tokens)
    segments_ids = [0 for i in range(len(indexed_tokens))]
    # print(segments_ids)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    # print(torch.Tensor(encoded_layers))
    # print(len(encoded_layers))
    # print(len(encoded_layers[0]))
    # print(len(encoded_layers[0][0]))
    # print(len(encoded_layers[0][0][0]))
    output = encoded_layers[11][0]
    output = output[1:output.size(0)-1]
    # print(len(output))
    # print(len(output[0]))
    # print(output[0])

    return output, tokenized_text


if __name__ == "__main__":
    # Test

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # text = "I love you."
    # text = add_cls_sep(text)
    # print(text)
    # tokenized_text = tokenizer.tokenize(text)
    # print(tokenized_text)
    # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # print(indexed_tokens)
    # segments_ids = [0 for i in range(len(indexed_tokens))]
    # print(segments_ids)
    # tokens_tensor = torch.tensor([indexed_tokens])
    # segments_tensors = torch.tensor([segments_ids])

    # with torch.no_grad():
    #     encoded_layers, _ = model(tokens_tensor, segments_tensors)

    # # print(torch.Tensor(encoded_layers))
    # print(len(encoded_layers))
    # print(len(encoded_layers[0]))
    # print(len(encoded_layers[0][0]))
    # print(len(encoded_layers[0][0][0]))
    # output = encoded_layers[11][0]
    # print(len(output))
    # print(len(output[0]))
    # print(output[0])

    output = get_bert_embedding("I love you.", model, tokenizer)
    print(len(output))
    print(len(output[0]))
