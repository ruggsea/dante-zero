import os
import sys
import io
import contextlib
import signal

# Add Dante module to path and import it
try:
    # Save current directory
    current_dir = os.getcwd()
    
    # Add Dante directory to Python path
    dante_dir = os.path.join(current_dir, 'Dante')
    sys.path.insert(0, dante_dir)
    
    # Change to Dante directory to ensure dictionary is found
    os.chdir(dante_dir)
    
    # Import the module
    import dante
    
    # Change back to original directory
    os.chdir(current_dir)
    
    # Patch the input function to avoid hanging
    original_input = input
    def patched_input(*args, **kwargs):
        return ""
    
    # Replace the input function in the dante module
    dante.input = patched_input
    
except Exception as e:
    print(f"Error: Dante module not available. Error: {e}")
    sys.exit(1)

def is_endecasillabo(verse):
    """
    Check if a verse is an endecasillabo using the Dante module.
    
    Args:
        verse (str): The verse to check
        
    Returns:
        tuple: (is_valid, syllabification) where:
            - is_valid (bool): True if the verse is a valid endecasillabo
            - syllabification (str): The syllabification of the verse if valid, or error message if not
    """
    try:
        # Set dante's verbose level to 0 to avoid messages
        original_verbose = dante.verbose
        dante.verbose = 0
        
        # Redirect stdout to suppress any print statements
        with contextlib.redirect_stdout(io.StringIO()):
            # Process the verse with verbose=0
            is_valid, syllabification = dante.process_verse(verse, verbose=0)
        
        # Restore original verbose level
        dante.verbose = original_verbose
        
        return is_valid, syllabification
    except Exception as e:
        return False, f"Error: {str(e)}"

def _enumerate_states(verse):
    """
    Enumerate ALL syllabification states produced by the Dante module for a verse.
    Returns a list of states as defined in Dante:
      state = (syllabs: str, prob: float, tot: int, pr: int, checks: tuple[bool,bool,bool])
    Only states that satisfy the 10th-syllable constraint (check10) are kept,
    mirroring Dante's internal filtering.
    """
    # Silence Dante internals
    original_verbose = dante.verbose
    dante.verbose = 0
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            # Tokenize like Dante.process_verse → process_tokenized_verse
            x = dante.preprocess(verse)
            xtokens = []
            for w in x:
                upper = w[0].isupper()
                tokens = dante.get_info(w)
                ctokens = [((pl, n, a, pr), dante.capital(upper, ws), prob) for ((pl, n, a, pr), ws, prob) in tokens]
                xtokens.append(ctokens)

            # Build all states (replicates check_verse without collapsing to best)
            states = [("", 1, 0, 0, (False, False, False))]
            for tokens in xtokens:
                states = dante.extend_multiple(states, tokens)

            # Keep only states that pass the metric 10th-syllable check
            states = [s for s in states if dante.check10(s)]
            if not states:
                return []

            # Prefer admissible (accent at 4th or 6th) else fall back to anomalous
            admissible, anomalous = dante.split(states)
            use_me = admissible if admissible else anomalous

            # Highest probability first
            use_me.sort(reverse=True, key=lambda s: s[1])
            return use_me
    finally:
        dante.verbose = original_verbose

def get_all_syllabifications(verse, max_candidates=None):
    """
    Return all syllabification options with probabilities.
    
    Args:
        verse (str): input line
        max_candidates (int|None): if set, truncate to top-K by probability
    
    Returns:
        (is_valid: bool, options: list[dict]) where each option has:
          - 'syllabification': str
          - 'probability': float (relative, not normalized)
          - 'syllable_count': int
          - 'checks': tuple(bool,bool,bool)  # accent checks @ 4th/6th/10th
    """
    states = _enumerate_states(verse)
    if not states:
        return False, []

    selected = states[:max_candidates] if max_candidates else states
    options = []
    for (syllabs, prob, tot, _pr, checks) in selected:
        options.append({
            "syllabification": syllabs,
            "probability": float(prob),
            "syllable_count": int(tot),
            "checks": checks,
        })
    return True, options

def safe_get_all_syllabifications(verse, timeout_seconds=1, max_candidates=None):
    """
    Timeout-safe wrapper around get_all_syllabifications.
    Returns (False, []) on timeout or error.
    """
    def handler(signum, frame):
        raise TimeoutError("syllabification timeout")

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout_seconds)
        return get_all_syllabifications(verse, max_candidates=max_candidates)
    except Exception:
        return False, []
    finally:
        signal.alarm(0)

# Simple test function
def test():
    # Test with a valid endecasillabo from Dante
    verse = "Nel mezzo del cammin di nostra vita"
    is_valid, syllabification = is_endecasillabo(verse)
    print(f"Verse: '{verse}'")
    print(f"Is Endecasillabo: {is_valid}")
    print(f"Syllabification: {syllabification}")
    
    # Test with an invalid verse
    verse = "This is not an Italian verse at all"
    is_valid, syllabification = is_endecasillabo(verse)
    print(f"\nVerse: '{verse}'")
    print(f"Is Endecasillabo: {is_valid}")
    print(f"Syllabification: {syllabification}")
    

if __name__ == "__main__":
    test() 