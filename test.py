import torch
from torchvision import transforms
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from utils.vocab import Vocabulary
from PIL import Image
import argparse

def load_image(image_path, transfrom = None):
    image = Image.open(image_path).convert("RGB")
    if transfrom:
        image = transfrom(image).unsqueeze(0)
    return image

def generate_caption(encoder, decoder, vocab, image_path, device, max_length=20):
    """
        Genreate caption using trained model
    """
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = load_image(image_path, transform).to(device)

    # Encode the image using train encoder
    with torch.no_grad():
        features = encoder(image)
    
    caption = ['<start>']
    input_word = torch.tensor([vocab('<start>')]).unsqueeze(0).to(device)

    # generate caption word by word
    for _ in range(max_length):
        with torch.no_grad():
            output = decoder(features, input_word)
            _, predicted = output.max(2) #get the word with maximum probability
            predicted_word_idx = predicted.item()
            word = vocab.idx2word[predicted_word_idx]
            caption.append(word)

            # if <end> token is generated then stop prediction
            if word == '<end>':
                break
            # prepare the next imput word

            input_word = torch.tensor([predicted_word_idx]).unsqueeze(0).to(device)

        return ' '.join(caption[1:-1]) # return caption ignoring start and end
    
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = EncoderCNN(args.embed_size).to(device)
    deocder = DecoderRNN(args.embed_size, args.hidden_size, args.vocab_size, num_layers = 1).to(device)

    encoder.load_state_dict(torch.load(args.encoder_path, map_location = device))
    decoder.load_state_dict(torch.load(args.decoder_path, map_location = device))

    vocab = Vocabulary(args.vocab_path)

    caption = generate_caption(encoder, decoder, vocab, args.image_path, device)
    print(f"Generated caption: {caption}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the image captioning model.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--encoder_path', type=str, required=True, help='Path to the trained encoder model.')
    parser.add_argument('--decoder_path', type=str, required=True, help='Path to the trained decoder model.')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to the vocabulary object.')
    parser.add_argument('--embed_size', type=int, default=256, help='Embedding size.')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size of the LSTM.')
    parser.add_argument('--vocab_size', type=int, required=True, help='Size of the vocabulary.')

    args = parser.parse_args()
    main(args)


    
