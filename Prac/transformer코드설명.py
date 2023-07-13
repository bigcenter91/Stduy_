import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from util import clones
from transformers.activations import get_activation
"""
self-Attention의 경우 Query Q, Key K, Value V를 입력으로 받아
MatMul(Q,K) -> Scale -> Masking(opt. Decoder) -> Softmax -> MatMul(result, V)

"""

def self_attention(query, key, value, mask=None):
  key_transpose = torch.transpose(key,-2,-1)             # (batch, head_num, d_k, token_)
  #키의 차원 마지막 차원과 그 전의 차원을 바꾸겠다.(cf)Transpose는 그 안에 값도 다 바꾼다.,permute는 중간만.)
  print(key_transpose.shape)
  matmul_result = torch.matmul(query,key_transpose)      # MatMul(Q,K)
  #쿼리와 키의 행렬곱.
  d_k = key.size()[-1]
  #d_k는 key 텐서의 마지막 차원의 크기를 나타내는 변수입니다. 
  attention_score = matmul_result/math.sqrt(d_k)         # Scale
  # 행렬곱한거에 루트dk를 나누기.
  if mask is not None:
    attention_score = attention_score.masked_fill(mask == 0, -1e20)
  #이 부분은 mask가 주어진 경우, 어텐션 스코어에서 mask 값이 0인 위치에 -1e20을 채워줍니다.
  softmax_attention_score = F.softmax(attention_score,dim=-1)  # 어텐션 값
  # 어태인션 스코어에 나온 값들을 softmax를 취해줌.
  result = torch.matmul(softmax_attention_score,value)
  # 소프트맥스에서 나온 값들을 value값과 행렬곱.
  return result, softmax_attention_score
  # 결과값 출력.

"""
멀티헤드 어텐션
MultiHead(Q,K,V) = Concat(head_1,head_2,...head_n)W^O
head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
W^Q는 모델의 dimension x d_k
W^K는 모델의 dimension x d_k
W^V는 모델의 dimension x d_v
W^O는 d_v * head 갯수 x 모델 dimension
논문에서는 헤더의 갯수를 8개 사용
"""
class MultiHeadAttention(nn.Module): # MultiHeadAttention은 nn.Module을 상속하는 클래스입니다.
  def __init__(self, head_num =8 , d_model = 512,dropout = 0.1): #헤드개수 8, 모델차원512,드롭아웃 0.1
    super(MultiHeadAttention,self).__init__() # 매개변수와 모듈을 초기화

    self.head_num = head_num
    self.d_model = d_model
    self.d_k = self.d_v = d_model // head_num

    self.w_q = nn.Linear(d_model,d_model) #각 주의 헤드에 대한 키 및 값의 차원
    self.w_k = nn.Linear(d_model,d_model)
    self.w_v = nn.Linear(d_model,d_model)
    self.w_o = nn.Linear(d_model,d_model)

    self.self_attention = self_attention
    self.dropout = nn.Dropout(p=dropout)
    #여기 있는 값들 다 초기화
  
  def forward(self, query, key, value, mask = None):
    if mask is not None:
      mask = mask.unsqueeze(1)
      # 마스크가 제공되면 (batch_num, 1, seq_len, seq_len) 형태로 확장
    batche_num = query.size(0)
    # batch_num은 첫 번째 차원을 따라 쿼리 텐서의 크기로 설정
    query = self.w_q(query).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)
    key = self.w_k(key).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)
    value = self.w_v(value).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)
    #  'query' 임베딩에 대해 선형 변환을 수행하고 어텐션 헤드를 분리하도록 재구성하고, 
    # 멀티 헤드 어텐션 메커니즘에서 추가 처리를 위해 차원을 올바르게 정렬하도록 텐서
    # view() 메서드는 텐서를 재구성
    # 한마디로 차원 정렬
    attention_result, attention_score = self.self_attention(query, key, value, mask)
    # 어텐션 결과 및 어텐션 점수를 계산하기 위해 변환된 쿼리, 키, 값 및 마스크와 함께 호출
    attention_result = attention_result.transpose(1,2).contiguous().view(batche_num, -1, self.head_num * self.d_k)
    #어텐션 결과는 (batch_num, seq_len, head_num * d_k)로 재구성된 후 선형 레이어 self.w_o를 거쳐 반환

    return self.w_o(attention_result)

"""
Position-wise Feed-Forward Networks
FFN(x) = max(0,xW_1 + b_1)W_2+b2
입력과 출력은 모두 d_model의 dimension을 가지고
내부의 레이어는 d_model * 4의 dimension을 가진다.
"""
class FeedForward(nn.Module):
  def __init__(self,d_model, dropout = 0.1):
    super(FeedForward,self).__init__()
    self.w_1 = nn.Linear(d_model, d_model*4)
    self.w_2 = nn.Linear(d_model*4, d_model)
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x):
    return self.w_2(self.dropout(F.relu(self.w_1(x))))
"""
Layer Normalization
: layer의 hidden unit들에 대해서 mean과 variance를 구한다. 
nn.Parameter는 모듈 파라미터로 여겨지는 텐서
"""
class LayerNorm(nn.Module):
  def __init__(self, features, eps=1e-6): #'features'는 입력 기능 차원의 크기, #eps : 0방지
    super(LayerNorm,self).__init__() #초기화
    self.a_2 = nn.Parameter(torch.ones(features)) #학습 가능한 스타일 매개변수
    self.b_2 = nn.Parameter(torch.zeros(features)) #학습 가능한 이동 매개변수
    self.eps = eps
  def forward(self, x): #'forward' 메서드는 레이어 정규화 작업의 정방향 패스를 수행
    mean = x.mean(-1, keepdim =True) 
    # mean은 마지막 차원(-1)을 따라 입력 텐서의 평균을 계산하고 차원을 유지
    std = x.std(-1, keepdim=True)    
    # 표준편차은 마지막 차원(-1)을 따라 입력 텐서의 평균을 계산하고 차원을 유지

    return self.a_2 * (x-mean)/ (std + self.eps) + self.b_2

class ResidualConnection(nn.Module):
  def __init__(self, size, dropout):
    super(ResidualConnection,self).__init__()
    self.norm = LayerNorm(size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, sublayer):
    return x + self.dropout((sublayer(self.norm(x))))
  #sublayer(더 큰 모델 내의 하위 계층을 나타내는 콜러블)의 두 가지 인수
"""
Encoder 블록은 FeedForward 레이어와 MultiHead 어텐션 레이어를 가진다.
"""
class Encoder(nn.Module):
  def __init__(self, d_model, head_num, dropout):
    super(Encoder,self).__init__() #인수 초기화.
    self.multi_head_attention = MultiHeadAttention(d_model= d_model, head_num= head_num)
    # MultiHeadAttention 클래스는 멀티 헤드 주의 작업을 수행
    self.residual_1 = ResidualConnection(d_model,dropout=dropout)\
    # self.residual_1 인스턴스는 ResidualConnection 클래스를 인스턴스화하여 생성
    self.feed_forward = FeedForward(d_model)
    # ResidualConnection 클래스는 잔류 연결 메커니즘
    self.residual_2 = ResidualConnection(d_model,dropout=dropout)
    # 입력 차원 및 드롭아웃 비율
  def forward(self, input, mask):
    x = self.residual_1(input, lambda x: self.multi_head_attention(x, x, x, mask)) #람다함수로 쿼리,키,벨류 마스크, 설정
    x = self.residual_2(x, lambda x: self.feed_forward(x))
    return x
    # self.feed_forward 인스턴스는 FeedForward 클래스를 인스턴스화하여 생성

# 한마디로 멀티 레지듀얼 피드포워드 레지듀얼 이렇게 모델 불러와서 인코더과정을 수행.

"""
Decoder 블록은 FeedForward 레이어와 MultiHead 어텐션, Masked Multihead 어텐션 레이어를 가진다.
MaskedMultiHeadAttention -> MultiHeadAttention(encoder-decoder attention) -> FeedForward
"""

class Decoder(nn.Module):
  def __init__(self, d_model,head_num, dropout):
    super(Decoder,self).__init__()
    self.masked_multi_head_attention = MultiHeadAttention(d_model= d_model, head_num= head_num)
    self.residual_1 = ResidualConnection(d_model,dropout=dropout)

    self.encoder_decoder_attention = MultiHeadAttention(d_model= d_model, head_num= head_num)
    self.residual_2 = ResidualConnection(d_model,dropout=dropout)

    self.feed_forward= FeedForward(d_model)
    self.residual_3 = ResidualConnection(d_model,dropout=dropout)


  def forward(self, target, encoder_output, target_mask, encoder_mask):
    # target, x, target_mask, input_mask
    x = self.residual_1(target, lambda x: self.masked_multi_head_attention(x, x, x, target_mask))
    x = self.residual_2(x, lambda x: self.encoder_decoder_attention(x, encoder_output, encoder_output, encoder_mask))
    x = self.residual_3(x, self.feed_forward)

    return x

# 한마디로 멀티 레지듀얼 피드포워드 레지듀얼 이렇게 모델 불러와서 디코더과정을 수행.

class Embeddings(nn.Module):
  def __init__(self, vocab_num, d_model):
    super(Embeddings,self).__init__()
    self.emb = nn.Embedding(vocab_num,d_model)
    self.d_model = d_model
  def forward(self, x):
    """
    1) 임베딩 값에 math.sqrt(self.d_model)을 곱해주는 이유는 무엇인지 찾아볼것
    2) nn.Embedding에 다시 한번 찾아볼것
    """
    return self.emb(x) * math.sqrt(self.d_model)
"""
Positional Encoding
트랜스포머는 RNN이나 CNN을 사용하지 않기 때문에 입력에 순서 값을 반영해줘야 한다.
예) 나는 어제의 오늘
PE (pos,2i) = sin(pos/10000^(2i/d_model))
PE (pos,2i+1) = cos(pos/10000^(2i/d_model)) 
"""
class PositionalEncoding(nn.Module): # PositionalEncoding은 nn.Module을 상속하는 클래스입니다. 
  def __init__(self, max_seq_len, d_model,dropout=0.1):
    # max_seq_len은 시퀀스의 최대 길이, d_model은 모델의 차원 수, dropout은 드롭아웃 나타냅니다. 
    super(PositionalEncoding,self).__init__()
    self.dropout = nn.Dropout(p=dropout)
    # dropout은 nn.Dropout을 초기화하여 드롭아웃 레이어로 사용
    pe = torch.zeros(max_seq_len, d_model)
    # pe는 최대 시퀀스 길이와 모델 차원으로 구성된 0으로 초기화된 텐서
    position = torch.arange(0,max_seq_len).unsqueeze(1)
    # position은 0부터 최대 시퀀스 길이까지의 값을 포함하는 열 벡터입니다.
    base = torch.ones(d_model//2).fill_(10000)
    # base는 모델 차원 수의 절반 크기로 구성된 10000으로 채워진 텐서
    pow_term = torch.arange(0, d_model, 2) / torch.tensor(d_model,dtype=torch.float32)
    # pow_term은 0부터 모델 차원 크기까지 2씩 증가하는 값을 나누어 얻은 텐서
    div_term = torch.pow(base,pow_term)
    # div_term은 base를 pow_term에 따라 거듭제곱한 텐서
    pe[:, 0::2] = torch.sin(position / div_term)
    pe[:, 1::2] = torch.cos(position / div_term)
    # sin은 짝수일때, cos은 홀수일때 사용.
    pe = pe.unsqueeze(0)
    # pe 텐서에 차원을 추가하여 (1, max_seq_len, d_model)
    self.register_buffer('positional_encoding', pe)
    # positional_encoding을 pe로 등록하여 학습되지 않는 변수로 설정
  
  def forward(self, x):
    x = x + Variable(self.positional_encoding[:, :x.size(1)], requires_grad=False)
    return self.dropout(x)
  #forward 함수는 입력 x에 위치 인코딩 값을 더한 후, 
  # 드롭아웃을 적용하여 결과를 반환합니다. 
  # self.positional_encoding[:, :x.size(1)]은 입력 x와 동일한 길이의 위치 인코딩 값을 선택합니다. 
  # 이를 x에 더하여 위치 정보를 모델 입력에 반영합니다.


class PositionalEmbedding(nn.Module):
  def __init__(self, dim, max_seq_len):
    super().__init__()
    self.embedding = nn.Embedding(max_seq_len, dim)

  def forward(self, x):
    t = torch.arange(x.shape[1], device=x.device)
    return self.embedding(t)

class Generator(nn.Module):
  def __init__(self, d_model, vocab_num):
    super(Generator, self).__init__()
    self.proj_1 = nn.Linear(d_model, d_model*4)
    self.proj_2 = nn.Linear(d_model*4, vocab_num)

  def forward(self, x):
    x = self.proj_1(x)
    x = self.proj_2(x)
    return x
class Transformer(nn.Module):
  def __init__(self,vocab_num, d_model, max_seq_len, head_num, dropout, N):
    super(Transformer,self).__init__()
    self.embedding = Embeddings(vocab_num, d_model)
    self.positional_encoding = PositionalEncoding(max_seq_len,d_model)

    self.encoders = clones(Encoder(d_model=d_model, head_num=head_num, dropout=dropout), N)
    self.decoders = clones(Decoder(d_model=d_model, head_num=head_num, dropout=dropout), N)

    self.generator = Generator(d_model, vocab_num)

  def forward(self, input, target, input_mask, target_mask, labels=None):
      x = self.positional_encoding(self.embedding(input))
      for encoder in self.encoders:
        x = encoder(x, input_mask)

      target = self.positional_encoding(self.embedding(target))
      for decoder in self.decoders:
        # target, encoder_output, target_mask, encoder_mask)
        target = decoder(target, x, target_mask, input_mask)

      lm_logits = self.generator(target)
      loss = None
      if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=0)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

      return lm_logits, loss
  def encode(self,input, input_mask):
    x = self.positional_encoding(self.embedding(input))
    for encoder in self.encoders:
      x = encoder(x, input_mask)
    return x

  def decode(self, encode_output, encoder_mask, target, target_mask):
    target = self.positional_encoding(self.embedding(target))
    for decoder in self.decoders:
      #target, encoder_output, target_mask, encoder_mask
      target = decoder(target, encode_output, target_mask, encoder_mask)

    lm_logits = self.generator(target)

    return lm_logits
class TransformerMRCHead(nn.Module):
  def __init__(self, dim, num_labels,hidden_dropout_prob=0.3):
    super().__init__()
    self.dense = nn.Linear(dim, 1*dim)
    self.dropout = nn.Dropout(hidden_dropout_prob)
    self.out_proj = nn.Linear(1*dim,num_labels)

  def forward(self, x, **kwargs):
    # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
    x = self.dropout(x)
    x = self.dense(x)
    x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
    x = self.dropout(x)
    x = self.out_proj(x)
    return x

class TransformerMRCModel(nn.Module):
  def __init__(self, vocab_size, dim, depth, max_seq_len, head_num, num_labels=2, causal=False, dropout_prob=0.2):
    super().__init__()
    self.transformer = TransformerLM(
      vocab_size=vocab_size,
      dim=dim,
      depth=depth,
      max_seq_len=max_seq_len,
      head_num=head_num,
    )
    self.mrc_head = TransformerMRCHead(dim, num_labels)

  def forward(self,
              input_ids=None,
              input_mask=None,
              start_positions=None,
              end_positions=None,
              **kwargs):
    # 1. transformer의 출력
    _, outputs = self.transformer(input_ids, input_mask)

    # 2. mrc를 위한
    logits = self.mrc_head(outputs)

    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)

    if start_positions is not None and end_positions is not None:
      # If we are on multi-GPU, split add a dimension
      if len(start_positions.size()) > 1:
        start_positions = start_positions.squeeze(-1)
      if len(end_positions.size()) > 1:
        end_positions = end_positions.squeeze(-1)
      # sometimes the start/end positions are outside our model inputs, we ignore these terms
      ignored_index = start_logits.size(1)
      start_positions.clamp_(0, ignored_index)
      end_positions.clamp_(0, ignored_index)

      loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
      start_loss = loss_fct(start_logits, start_positions)
      end_loss = loss_fct(end_logits, end_positions)
      total_loss = (start_loss + end_loss) / 2
      return total_loss
    else:
      return start_logits, end_logits

class TransformerLM(nn.Module):
  def __init__(self, vocab_size, dim=512,  depth= 12, max_seq_len=512, head_num=8, dropout= 0.1):
    super(TransformerLM,self).__init__()

    self.token_emb= nn.Embedding(vocab_size, dim)
    self.position_emb = PositionalEmbedding(dim,max_seq_len)
    self.encoders = clones(Encoder(d_model=dim, head_num=head_num, dropout=dropout), depth)
    self.norm = nn.LayerNorm(dim)
    self.lm_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Linear(dim, vocab_size)
            )

  def forward(self, input_ids, input_mask):
    x = self.token_emb(input_ids)
    x = x + self.position_emb(input_ids).type_as(x)

    for encoder in self.encoders:
      x = encoder(x, input_mask)
    x = self.norm(x)

    return self.lm_head(x), x  # lm_head, performer_embedding



if __name__=="__main__":
  pass
