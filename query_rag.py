"""
Query script for the DesignSafe & SimCenter Documentation RAG system
Searches across:
- DesignSafe User Guide & Training
- SimCenter Tools (quoFEM, EE-UQ, Hydro-UQ, WE-UQ, PBE, R2D)
"""

import os
from dotenv import load_dotenv
from create_rag import DesignSafeRAG, DOC_SOURCES

# Load environment variables
load_dotenv()


def main():
    """Main function to query the RAG system."""
    # Initialize RAG system
    rag = DesignSafeRAG()

    print("\n" + "=" * 60)
    print("DesignSafe & SimCenter Documentation RAG Query System")
    print("=" * 60)

    print("\nSearches across:")
    print("  DesignSafe:")
    for key in ["designsafe-user-guide", "designsafe-training"]:
        if key in DOC_SOURCES:
            print(f"    - {DOC_SOURCES[key]['name']}")

    print("  SimCenter Tools:")
    for key in ["quofem", "ee-uq", "hydro-uq", "we-uq", "pbe", "r2d"]:
        if key in DOC_SOURCES:
            print(f"    - {DOC_SOURCES[key]['name']}")

    print("\nType 'quit' or 'exit' to stop")
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
            response = rag.query(query, top_k=8)

            # Display response
            print("Answer:")
            print("-" * 50)
            print(response.response)
            print()

            # Display source documents
            if hasattr(response, 'source_nodes'):
                print("Sources:")
                print("-" * 50)
                for i, node in enumerate(response.source_nodes, 1):
                    metadata = node.metadata
                    source_name = metadata.get('source', 'Unknown')
                    title = metadata.get('title', 'N/A')
                    section = metadata.get('section', 'N/A')
                    url = metadata.get('url', 'N/A')

                    print(f"{i}. [{source_name}] {title}")
                    print(f"   Section: {section}")
                    print(f"   URL: {url}")
                    print()

        except Exception as e:
            print(f"Error: {e}")
            print()


if __name__ == "__main__":
    main()
