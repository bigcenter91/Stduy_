import openai

openai.api_key='my key'

def generate_text(prompt,model:str='gpt-4')->str:
    return openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=100, # 최대 토큰 수 설정
            n=1, # 생성할 응답 수 설정
            stop=None, # 종료 문자 설정(예: "\n")
            temperature=0.7, # 생성된 텍스트 다양성 조절(낮은 값: 더 일관된 텍스트, 높은 값: 더 다양한 텍스트)
            top_p=1,
    ).choice[0].text.strip()

conversation_history = ""

while True:
    user_input = input("User: ")
    conversation_history += f"User: {user_input}\n"

    # 대화 내용을 프롬프트로 사용
    prompt = f"{conversation_history}Assistant: "
    generated_text = generate_text(prompt)
    print(f"Assistant: {generated_text}")

    conversation_history += f"Assistant: {generated_text}\n"