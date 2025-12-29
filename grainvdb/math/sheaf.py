import numpy as np

def compute_gluing_energy(vec1, vec2, context_vectors_1, context_vectors_2):
    """
    Simplified Sheaf Theory 'Gluing Energy' Proof.
    Measures the obstruction to consistency between two local sections.
    """
    # In a real sheaf, we'd use the Çech coboundary δ
    # Here we simulate it: how well do the 'neighborhoods' of the results align?
    
    # alignment = mean similarity of neighborhood 1 to neighborhood 2
    # If high, they glue (consistent context). If low, fracture.
    
    alignment = 0
    for v in context_vectors_1:
        alignment += np.max([np.dot(v, c2) for c2 in context_vectors_2])
    
    energy = 1.0 - (alignment / len(context_vectors_1))
    return energy

def run_sheaf_demo():
    print("--- Sheaf Theory vs. Flat KNN Graph RAG ---")
    
    # 1. Setup ambiguous vectors
    # Jaguar (Animal) and Jaguar (Car) are close in embedding space (token match)
    jaguar_animal = np.array([0.8, 0.1, 0.0])
    jaguar_car = np.array([0.75, 0.15, 0.0]) # Very close to each other
    
    # Contexts
    animal_neighborhood = [
        np.array([0.9, 0.0, 0.0]), # Feline
        np.array([0.85, 0.05, 0.0]), # Spotted
        np.array([0.8, -0.1, 0.0])  # Jungle
    ]
    
    car_neighborhood = [
        np.array([0.0, 0.9, 0.0]), # Engine
        np.array([0.1, 0.85, 0.0]), # Luxury
        np.array([0.2, 0.8, 0.0])   # V8
    ]
    
    print(f"Flat Similarity (Animal-Jag vs Car-Jag): {np.dot(jaguar_animal, jaguar_car):.4f}")
    
    # Compute Gluing Energy
    energy_same = compute_gluing_energy(jaguar_animal, jaguar_animal, animal_neighborhood, animal_neighborhood)
    energy_diff = compute_gluing_energy(jaguar_animal, jaguar_car, animal_neighborhood, car_neighborhood)
    
    print(f"\nGluing Energy (Consistent Context): {energy_same:.4f}")
    print(f"Gluing Energy (Fractured Context): {energy_diff:.4f}")
    
    print("\nRESULT:")
    if energy_diff > 0.5:
        print("✅ Sheaf Theory detected the REGIME FRACTURE (Context Mix).")
        print("   Flat KNN would have hallucinated a jaguar-car hybrid.")
    else:
        print("❌ Failed to detect fracture.")

if __name__ == "__main__":
    run_sheaf_demo()
