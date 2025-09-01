"""
Divine Comedy Verse Checker

This module provides a function to check if a verse is from Dante's Divine Comedy
by comparing it against a list of verses from Inferno, Purgatorio, and Paradiso.
"""

import os
import re
import string
from difflib import SequenceMatcher
import unicodedata
import random
import difflib
import hashlib
from collections import defaultdict

class DivineComedyChecker:
    def __init__(self):
        self.divine_comedy_verses = []
        self.divine_comedy_verses_set = set()  # For faster exact matching
        self.loaded = False
        self.similarity_threshold = 0.85  # Threshold for considering a verse similar enough
        self.cache = {}  # Cache for previously checked verses
        
        # Hash-based matching
        self.hash_table = defaultdict(list)  # Maps hash values to verses
        self.ngram_size = 3  # Size of character n-grams for hashing
        self.min_hash_functions = 5  # Number of hash functions to use
        self.hash_seeds = [42, 101, 199, 317, 631]  # Seeds for hash functions
    
    def get_all_verses(self):
        """
        Get all available verses from the Divine Comedy.
        
        Returns:
            list: All verses from the Divine Comedy
        """
        if not self.loaded:
            success = self.load_verses()
            if not success:
                return []
        
        return self.divine_comedy_verses.copy()
    
    def _compute_verse_hashes(self, verse):
        """
        Compute multiple hash values for a verse using different hash functions.
        
        Args:
            verse (str): The normalized verse to hash
            
        Returns:
            list: List of hash values
        """
        hashes = []
        for seed in self.hash_seeds[:self.min_hash_functions]:
            # Create a hash object with the seed
            h = hashlib.md5(f"{seed}".encode())
            # Update with the verse
            h.update(verse.encode())
            # Get the hash value as an integer
            hash_value = int(h.hexdigest(), 16) % (2**32)
            hashes.append(hash_value)
        return hashes
    
    def _compute_ngram_hashes(self, verse):
        """
        Compute hashes of character n-grams in the verse.
        
        Args:
            verse (str): The normalized verse to hash
            
        Returns:
            set: Set of hash values for n-grams
        """
        ngram_hashes = set()
        # Skip if verse is too short
        if len(verse) < self.ngram_size:
            return ngram_hashes
            
        # Generate character n-grams and hash them
        for i in range(len(verse) - self.ngram_size + 1):
            ngram = verse[i:i+self.ngram_size]
            # Use a simple hash function for n-grams
            ngram_hash = hash(ngram) % (2**32)
            ngram_hashes.add(ngram_hash)
            
        return ngram_hashes
    
    def load_verses(self, cantica_paths=None):
        """
        Load verses from the Divine Comedy.
        
        Args:
            cantica_paths (list): List of paths to the cantica files (Inferno, Purgatorio, Paradiso)
                                 If None, will try default paths
        
        Returns:
            bool: True if verses were loaded successfully, False otherwise
        """
        if self.loaded:
            return True
            
        if cantica_paths is None:
            # Default paths to try
            cantica_paths = [
                os.path.join('Dante', 'inferno.txt'),
                os.path.join('Dante', 'purgatorio.txt'),
                os.path.join('Dante', 'paradiso.txt')
            ]
            
        verses = []
        
        for path in cantica_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Split into lines
                    lines = content.split('\n')
                    
                    # Add non-empty lines
                    verses.extend([line.strip() for line in lines if line.strip()])
            except Exception as e:
                print(f"Error loading {path}: {e}")
                
        if not verses:
            return False
            
        self.divine_comedy_verses = verses
        
        # Create a set of normalized verses for faster exact matching
        self.divine_comedy_verses_set = set(self._normalize_verse(verse) for verse in verses)
        
        # Build hash tables for faster matching
        self._build_hash_tables()
        
        self.loaded = True
        return True
    
    def _build_hash_tables(self):
        """
        Build hash tables for all verses in the Divine Comedy.
        """
        # Clear existing hash tables
        self.hash_table = defaultdict(list)
        
        # Process each verse
        for verse in self.divine_comedy_verses:
            normalized = self._normalize_verse(verse)
            
            # Compute verse hashes
            verse_hashes = self._compute_verse_hashes(normalized)
            
            # Add verse to hash table for each hash value
            for h in verse_hashes:
                self.hash_table[h].append((normalized, verse))
                
        # Print stats
        print(f"Built hash tables with {len(self.hash_table)} unique hash values")
        print(f"Average bucket size: {sum(len(bucket) for bucket in self.hash_table.values()) / max(1, len(self.hash_table)):.2f}")
    
    def _normalize_verse(self, verse):
        """
        Normalize a verse for comparison (Unicode normalization, lowercase, remove punctuation).
        
        Args:
            verse (str): The verse to normalize
            
        Returns:
            str: The normalized verse
        """
        # Apply Unicode normalization (NFKD form)
        verse = unicodedata.normalize('NFKD', verse)
        
        # Convert to lowercase
        verse = verse.lower()
        
        # Remove punctuation and non-alphanumeric characters
        verse = re.sub(r'[^\w\s]', '', verse)
        
        # Remove extra whitespace
        verse = ' '.join(verse.split())
        
        return verse
    
    def _calculate_similarity(self, verse1, verse2):
        """
        Calculate the similarity between two verses.
        
        Args:
            verse1 (str): First verse
            verse2 (str): Second verse
            
        Returns:
            float: Similarity score between 0 and 1
        """
        return SequenceMatcher(None, verse1, verse2).ratio()
    
    def _get_bigrams(self, text):
        """
        Extract bigrams from a text.
        
        Args:
            text (str): The text to extract bigrams from
            
        Returns:
            list: List of bigrams
        """
        words = text.split()
        return [' '.join(words[i:i+2]) for i in range(len(words)-1)]
    
    def check_bigram_overlap(self, verse, threshold=0.5):
        """
        Check if a verse has significant bigram overlap with any verse from the Divine Comedy.
        
        Args:
            verse (str): The verse to check
            threshold (float): The minimum overlap ratio to consider as a match
            
        Returns:
            tuple: (has_overlap, details) where:
                - has_overlap (bool): True if significant bigram overlap is found
                - details (dict): Additional information about the match
        """
        if not self.loaded:
            success = self.load_verses()
            if not success:
                return False, {"error": "Failed to load Divine Comedy verses"}
        
        # Normalize the input verse
        normalized_verse = self._normalize_verse(verse)
        
        # Extract bigrams from the input verse
        verse_bigrams = set(self._get_bigrams(normalized_verse))
        
        if not verse_bigrams:
            return False, {
                "match_type": "none",
                "reason": "No bigrams in input verse",
                "original_verse": verse
            }
        
        # Find the verse with the highest bigram overlap
        best_match = None
        best_overlap_ratio = 0
        best_common_bigrams = set()
        
        for dc_verse in self.divine_comedy_verses:
            dc_bigrams = set(self._get_bigrams(dc_verse))
            if not dc_bigrams:
                continue
                
            common_bigrams = verse_bigrams.intersection(dc_bigrams)
            
            # Calculate overlap ratio (Jaccard similarity)
            overlap_ratio = len(common_bigrams) / len(verse_bigrams.union(dc_bigrams))
            
            if overlap_ratio > best_overlap_ratio:
                best_overlap_ratio = overlap_ratio
                best_match = dc_verse
                best_common_bigrams = common_bigrams
        
        has_overlap = best_overlap_ratio >= threshold
        
        return has_overlap, {
            "match_type": "bigram_overlap" if has_overlap else "none",
            "overlap_ratio": best_overlap_ratio,
            "best_match": best_match,
            "common_bigrams": list(best_common_bigrams),
            "original_verse": verse,
            "normalized_verse": normalized_verse,
            "threshold": threshold
        }
    
    def is_from_divine_comedy(self, verse, exact_match=True, verbose=False):
        """
        Check if a verse is from the Divine Comedy.
        
        Args:
            verse (str): The verse to check
            exact_match (bool): Whether to check for exact matches only
            verbose (bool): Whether to print verbose output
            
        Returns:
            tuple: (is_match, details) where:
                - is_match (bool): True if the verse is from the Divine Comedy
                - details (dict): Additional information about the match
        """
        # Skip very short verses
        if len(verse.strip()) < 10:
            return False, {"error": "Verse too short"}
            
        # Check cache first
        cache_key = f"{verse}_{exact_match}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        if not self.loaded:
            success = self.load_verses()
            if not success:
                return False, {"error": "Failed to load Divine Comedy verses"}
        
        # Normalize the input verse
        normalized_verse = self._normalize_verse(verse)
        
        # Check for exact match
        if exact_match:
            # Fast check using set
            is_match = normalized_verse in self.divine_comedy_verses_set
            
            if verbose:
                print(f"Checking if verse is from Divine Comedy (exact match): {is_match}")
            
            result = (is_match, {
                "match_type": "exact" if is_match else "none",
                "original_verse": verse,
                "normalized_verse": normalized_verse
            })
            
            # Cache the result
            self.cache[cache_key] = result
            return result
        
        # For similar matches, use hash-based approach
        # Compute hashes for the input verse
        verse_hashes = self._compute_verse_hashes(normalized_verse)
        
        # Collect candidate verses from hash table
        candidates = set()
        for h in verse_hashes:
            if h in self.hash_table:
                for norm_verse, orig_verse in self.hash_table[h]:
                    candidates.add((norm_verse, orig_verse))
        
        # If we have too few candidates, add some random verses
        if len(candidates) < 10:
            # Add some random verses to check (up to 100)
            sample_size = min(100, len(self.divine_comedy_verses))
            random_verses = random.sample(self.divine_comedy_verses, sample_size)
            for orig_verse in random_verses:
                norm_verse = self._normalize_verse(orig_verse)
                candidates.add((norm_verse, orig_verse))
        
        # Limit the number of candidates to check (max 200)
        if len(candidates) > 200:
            candidates = random.sample(list(candidates), 200)
        
        # Check for similar match among candidates
        best_match = None
        best_similarity = 0
        best_orig_verse = None
        
        for norm_verse, orig_verse in candidates:
            similarity = self._calculate_similarity(normalized_verse, norm_verse)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = norm_verse
                best_orig_verse = orig_verse
                
            # Early stopping if we find a very good match
            if similarity > 0.95:
                break
        
        is_match = best_similarity >= self.similarity_threshold
        
        if verbose:
            print(f"Best match similarity: {best_similarity:.2f}")
            print(f"Best matching verse: {best_match}")
            print(f"Is from Divine Comedy: {is_match}")
        
        result = (is_match, {
            "match_type": "similar" if is_match else "none",
            "similarity": best_similarity,
            "best_match": best_orig_verse,
            "original_verse": verse,
            "normalized_verse": normalized_verse,
            "threshold": self.similarity_threshold
        })
        
        # Cache the result
        self.cache[cache_key] = result
        return result

# Convenience function
def is_from_divine_comedy(verse, exact_match=True, verbose=False):
    """
    Check if a verse is from the Divine Comedy.
    
    Args:
        verse (str): The verse to check
        exact_match (bool): Whether to require an exact match or allow similar verses
        verbose (bool): Whether to print detailed information
        
    Returns:
        tuple: (is_from_divine_comedy, details) where:
            - is_from_divine_comedy (bool): True if the verse is from the Divine Comedy
            - details (dict or str): Details about the match or error message
    """
    try:
        checker = DivineComedyChecker()
        return checker.is_from_divine_comedy(verse, exact_match, verbose)
    except Exception as e:
        return False, f"Error: {str(e)}"

def test():
    """
    Test the Divine Comedy checker on sample verses.
    """
    checker = DivineComedyChecker()
    checker.load_verses()
    
    # Test with verses from the Divine Comedy
    divine_comedy_verses = [
        "Nel mezzo del cammin di nostra vita",
        "mi ritrovai per una selva oscura",
        "ché la diritta via era smarrita",
        "Tant'è amara che poco è più morte",
        "La gloria di colui che tutto move"
    ]
    
    # Test with verses not from the Divine Comedy
    non_divine_comedy_verses = [
        "This is not a verse from the Divine Comedy",
        "Questo non è un verso della Divina Commedia",
        "To be or not to be, that is the question",
        "Nel mezzo del cammin di nostra vita moderna",
        "mi ritrovai per una strada buia"
    ]
    
    # Track results
    real_verses_identified = 0
    real_verses_missed = 0
    fake_verses_identified = 0
    fake_verses_mistaken = 0
    
    print("\n=== Testing with verses from the Divine Comedy (exact match) ===")
    for verse in divine_comedy_verses:
        is_match, details = checker.is_from_divine_comedy(verse, exact_match=True, verbose=True)
        print(f"\nVerse: '{verse}'")
        print(f"Is from Divine Comedy: {is_match}")
        print(f"Match type: {details['match_type']}")
        print("-" * 50)
        if is_match:
            real_verses_identified += 1
        else:
            real_verses_missed += 1
    
    print("\n=== Testing with verses not from the Divine Comedy (exact match) ===")
    for verse in non_divine_comedy_verses:
        is_match, details = checker.is_from_divine_comedy(verse, exact_match=True, verbose=True)
        print(f"\nVerse: '{verse}'")
        print(f"Is from Divine Comedy: {is_match}")
        print(f"Match type: {details['match_type']}")
        print("-" * 50)
        if is_match:
            fake_verses_mistaken += 1
        else:
            fake_verses_identified += 1

    # Print results
    print("\n=== Test Results ===")
    print(f"Real verses correctly identified: {real_verses_identified} out of {len(divine_comedy_verses)}")
    print(f"Real verses missed: {real_verses_missed} out of {len(divine_comedy_verses)}")
    print(f"Fake verses correctly identified: {fake_verses_identified} out of {len(non_divine_comedy_verses)}")
    print(f"Fake verses mistakenly matched: {fake_verses_mistaken} out of {len(non_divine_comedy_verses)}")

if __name__ == "__main__":
    test() 