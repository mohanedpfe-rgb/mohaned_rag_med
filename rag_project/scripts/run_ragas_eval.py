import json
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall,
)
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from rag_project.app.rag_system import RAGSystem
from rag_project.configuration.settings import Settings

def main():
    print("Initializing RAGSystem...")
    settings = Settings.from_env()
    rag = RAGSystem(settings)
    
    # Load dataset
    data_path = Path("data/eval/v1/dataset.jsonl")
    if not data_path.exists():
        print(f"Dataset not found at {data_path}. Please create it first.")
        return
        
    questions = []
    ground_truths = []
    
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            questions.append(item["question"])
            ground_truths.append(item["acceptable_answers"])
            
    print(f"Loaded {len(questions)} questions.")
    
    answers = []
    contexts = []
    
    print("Generating answers...")
    for q in questions:
        # Use existing conversation history context
        answer, hits, _ = rag.answer(q)
        answers.append(answer)
        contexts.append([hit.text for hit in hits])
        
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    print("Running RAGAS evaluation...")
    
    # Configure Ollama for RAGAS
    llm = ChatOllama(model=settings.generation_model, base_url=settings.ollama_base_url)
    embeddings = OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)
    
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevance,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
    )
    
    print("\n=== RAGAS Evaluation Results ===")
    print(result)
    
    output_file = "ragas_report.json"
    result.to_pandas().to_json(output_file, orient="records", indent=2)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()
