import os
import sys
import random

# Import our simple endecasillabo checker
try:
    from simple_endecasillabo_checker import is_endecasillabo
    print("Successfully imported endecasillabo checker")
except Exception as e:
    print(f"Error importing endecasillabo checker: {e}")
    sys.exit(1)

def load_dante_verses(file_path, max_verses=50):
    """
    Load verses from a Dante text file.
    
    Args:
        file_path (str): Path to the text file
        max_verses (int): Maximum number of verses to load
        
    Returns:
        list: List of verses
    """
    verses = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and canto headers
                if line and not line.startswith('CANTO') and '•' not in line:
                    verses.append(line)
                    if len(verses) >= max_verses:
                        break
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        print("Checking current directory...")
        print(f"Current directory: {os.getcwd()}")
        print("Listing files in current directory:")
        print(os.listdir('.'))
        print("\nListing files in Dante directory:")
        try:
            print(os.listdir('./Dante'))
        except:
            print("Could not list Dante directory")
    return verses

def generate_non_dante_verses(count=20):
    """
    Generate non-Dante verses for testing.
    
    Args:
        count (int): Number of verses to generate
        
    Returns:
        list: List of non-Dante verses
    """
    non_dante_verses = [
        "This is not an Italian verse at all",
        "Questo è un verso troppo corto",
        "Questo verso è troppo lungo per essere considerato un endecasillabo dantesco",
        "Hello world, this is a test",
        "The quick brown fox jumps over the lazy dog",
        "To be or not to be, that is the question",
        "All that glitters is not gold",
        "A journey of a thousand miles begins with a single step",
        "La vita è bella ma non è un endecasillabo",
        "Roses are red, violets are blue",
        "I wandered lonely as a cloud",
        "Do not go gentle into that good night",
        "Water water everywhere nor any drop to drink",
        "Shall I compare thee to a summer's day",
        "Because I could not stop for Death",
        "Two roads diverged in a yellow wood",
        "The woods are lovely dark and deep",
        "Questo non è un verso di Dante Alighieri",
        "Oggi è una bella giornata di sole",
        "Il gatto miagola sul tetto della casa"
    ]
    return non_dante_verses[:count]

def test_dante_detector(verses, expected_valid=True, verbose=True):
    """
    Test the Dante detector on a list of verses.
    
    Args:
        verses (list): List of verses to test
        expected_valid (bool): Whether the verses are expected to be valid
        verbose (bool): Whether to print detailed results
        
    Returns:
        dict: Test results
    """
    results = {
        'total': len(verses),
        'valid': 0,
        'invalid': 0,
        'correct': 0,  # Count of correctly classified verses
        'accuracy': 0.0
    }
    
    for verse in verses:
        try:
            # Use our simple endecasillabo checker
            is_valid, syllabification = is_endecasillabo(verse)
            
            if is_valid:
                results['valid'] += 1
            else:
                results['invalid'] += 1
                
            # Calculate accuracy based on expected validity
            if is_valid == expected_valid:
                results['correct'] += 1
                
            if verbose:
                print(f"\nVerse: '{verse}'")
                print(f"Is Endecasillabo: {is_valid}")
                if is_valid:
                    print(f"Syllabification: {syllabification}")
        except Exception as e:
            print(f"Error processing verse: '{verse}'")
            print(f"Error: {e}")
            results['invalid'] += 1
            
            # If we expect invalid verses, count errors as correct classifications
            if not expected_valid:
                results['correct'] += 1
            
    # Calculate final accuracy
    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0.0
    
    return results

def main():
    # Load Dante verses from Inferno
    print("\n=== Loading Dante verses from Inferno ===")
    # Try different possible paths
    possible_paths = [
        os.path.join('Dante', 'inferno.txt'),
        'Dante/inferno.txt',
        './Dante/inferno.txt',
        '../Dante/inferno.txt',
        'inferno.txt'
    ]
    
    dante_verses = []
    for path in possible_paths:
        print(f"Trying path: {path}")
        verses = load_dante_verses(path, max_verses=30)
        if verses:
            dante_verses = verses
            print(f"Successfully loaded {len(verses)} verses from {path}")
            break
    
    if not dante_verses:
        print("Could not load Dante verses. Using sample verses instead.")
        dante_verses = [
            "Nel mezzo del cammin di nostra vita",
            "Mi ritrovai per una selva oscura",
            
            "Che la diritta via era smarrita",
            "Tant'è amara che poco è più morte",
            "Ma per trattar del ben ch'i' vi trovai",
            "Dirò de l'altre cose ch'i' v'ho scorte"
        ]
    
    # Generate non-Dante verses
    print("\n=== Generating non-Dante verses ===")
    non_dante_verses = generate_non_dante_verses(count=20)
    
    # Test on Dante verses (expected to be valid endecasillabi)
    print("\n=== Testing Dante detector on Dante verses (expected to be valid) ===")
    dante_results = test_dante_detector(dante_verses, expected_valid=True, verbose=True)
    
    # Test on non-Dante verses (expected to be invalid endecasillabi)
    print("\n=== Testing Dante detector on non-Dante verses (expected to be invalid) ===")
    non_dante_results = test_dante_detector(non_dante_verses, expected_valid=False, verbose=True)
    
    # Print summary
    print("\n=== Test Results Summary ===")
    print(f"Dante verses tested: {dante_results['total']}")
    print(f"Valid endecasillabi: {dante_results['valid']} ({dante_results['valid']/dante_results['total']*100:.2f}%)")
    print(f"Invalid endecasillabi: {dante_results['invalid']} ({dante_results['invalid']/dante_results['total']*100:.2f}%)")
    print(f"Accuracy on Dante verses: {dante_results['accuracy']*100:.2f}%")
    
    print(f"\nNon-Dante verses tested: {non_dante_results['total']}")
    print(f"Valid endecasillabi: {non_dante_results['valid']} ({non_dante_results['valid']/non_dante_results['total']*100:.2f}%)")
    print(f"Invalid endecasillabi: {non_dante_results['invalid']} ({non_dante_results['invalid']/non_dante_results['total']*100:.2f}%)")
    print(f"Accuracy on non-Dante verses: {non_dante_results['accuracy']*100:.2f}%")
    
    # Overall accuracy
    overall_accuracy = (dante_results['accuracy'] * dante_results['total'] + 
                        non_dante_results['accuracy'] * non_dante_results['total']) / (
                        dante_results['total'] + non_dante_results['total'])
    print(f"\nOverall accuracy: {overall_accuracy*100:.2f}%")

if __name__ == "__main__":
    main() 