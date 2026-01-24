"""
Query script for the DesignSafe User Guide RAG system
"""

import os
from dotenv import load_dotenv
from create_rag import DesignSafeRAG

# Load environment variables
load_dotenv()

def main():
    """Main function to query the RAG system."""
    # Initialize RAG system
    rag = DesignSafeRAG()
    
    print("DesignSafe User Guide RAG Query System")
    print("=" * 50)
    print("Type 'quit' or 'exit' to stop")
    print()
    
    while True:
        # Get user query
        query = input("Enter your question: ").strip()
        
        if query.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if not query:
            print("Please enter a valid question.")
            continue
        
        try:
            # Query the RAG system
            print("\nSearching...\n")
            response = rag.query(query, top_k=3)
            
            # Display response
            print("Answer:")
            print("-" * 40)
            print(response.response)
            print()
            
            # Display source documents
            if hasattr(response, 'source_nodes'):
                print("Sources:")
                print("-" * 40)
                for i, node in enumerate(response.source_nodes, 1):
                    metadata = node.metadata
                    print(f"{i}. {metadata.get('title', 'N/A')} - {metadata.get('section', 'N/A')}")
                    print(f"   URL: {metadata.get('url', 'N/A')}")
                    print()
            
        except Exception as e:
            print(f"Error: {e}")
            print()


if __name__ == "__main__":
    main()