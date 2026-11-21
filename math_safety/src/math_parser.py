import re
import ast
import regex
import sympy as sp


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", string)
    _string = re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", _string)
    return _string


def _fix_tan(string):
    _string = re.sub(r"\\tan(-?[0-9.a-zA-Z]+)", r"\\tan{\1}", string)
    _string = re.sub(r"\\tan\s+(\w+)$", r"\\tan{\1}", _string)
    return _string


def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")

    if string.startswith("\\text{") and string.endswith("}"):
        string = string.split("{", 1)[1][:-1]

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("cfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "").strip()
    string = string.replace("^\\circ", "").strip()

    string = regex.sub(r"\{(c|m)?m\}(\^(2|3))?", "", string).strip()
    string = regex.sub(r"p\.m\.$", "", string).strip()
    string = regex.sub(r"(\d)\s*t$", r"\1", string).strip()

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "%")
    string = string.replace("\%", "%")
    # string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    # string = string.replace("and", "")
    string = string.replace("\\mathbf", "")
    string = string.replace("\\mathrm", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace("\"", "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    string = _fix_sqrt(string)
    string = _fix_tan(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    string = regex.sub(r"(\\|,|\.)+$", "", string)

    return string


def extract_boxed_answers(text):
    answers = []
    for piece in text.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    # Return only the last boxed answer if there are multiple
    return [answers[-1]] if answers else []


def extract_last_dollar_expression(text):
    answer = []
    # convert double dollar signs to single dollar signs
    text = re.sub(r'\$\$(.*?)\$\$', r'$\1$', text, flags=re.DOTALL)
    # find the last expression between single dollar signs
    matches = re.findall(r'\$([^$]+)\$', text)
    if matches:
        answer.append(matches[-1])
        return answer
    else:
        return None


def extract_solutionis_dollar_expression(text):
    answer = []
    # convert double dollar signs to single dollar signs
    text = re.sub(r'\$\$(.*?)\$\$', r'$\1$', text, flags=re.DOTALL)
    # find the last expression between single dollar signs
    text = re.search(r'solutions.*?is\s+\$(.*?)\$', text, re.DOTALL)
    try:
        match = re.findall(r'\$([^$]+)\$', text)
        if match:
            answer.append(match.group(1))
            return answer
        else:
            return None
    except:
        return None


def parse_fraction(input_str):
    """
    Parses LaTeX-like fraction strings (including negatives) and evaluates them as float decimals.
    If execution exceeds the timeout limit, it raises an error.
    """
    cleaned_str = input_str.replace("\\dfrac", "").replace("\\frac", "")

    is_negative = cleaned_str.startswith("-")

    # Strip the negative sign for easier parsing
    if is_negative:
        cleaned_str = cleaned_str[1:]

    cleaned_str_lst = cleaned_str.strip("{}").split("}{")
    if len(cleaned_str_lst) != 2:
        return None
    else:
        numerator = cleaned_str_lst[0]
        denominator = cleaned_str_lst[1]
    if re.fullmatch(r"-?\d+", numerator):
        numerator = int(numerator)
    else:
        return None
    if re.fullmatch(r"-?\d+", denominator):
        denominator = int(denominator)
        if denominator == 0:
            return None
    else:
        return None

    result = numerator / denominator

    if is_negative:
        result = -result

    return str(float(result))


def handle_minor_cases(ans, in_the_form=False, separated_by_commas=False):
    # get rid of "~~~ is"
    ans = ans.split('is')[-1].strip()

    # remove \( answer \) -> check qwen 3b
    ans = re.sub(r'\\\((.*?)\\\)', r'\1', ans)

    # remove ** answer ** -> check gemma
    ans = re.sub(r'\*\*(.*?)\*\*', r'\1', ans)

    # remove ** in front and back -> check gemma
    ans = ans.lstrip("**")
    ans = ans.rstrip("**")

    # Extract the last single alphabet inside parentheses (A), (B), etc.
    matches = re.findall(r'\(([A-Za-z])\)', ans)
    if matches:
        ans = matches[-1]

    # Split words and check if all except the last are alphabetic (e.g. "degree by 42")
    words = ans.split()
    if len(words) > 1 and all(re.match(r"^[A-Za-z']+$", word) for word in words[:-1]):
        ans = words[-1]

    # Get rid of units: if last element is alphabetic and second-last is not
    list_temp = ans.split()
    if len(list_temp) > 1:
        if list_temp[-1].isalpha():
            if re.search(r'\d', list_temp[-2]):
                ans = " ".join(list_temp[:-1])  # Remove the last word (unit)
            elif len(list_temp) > 2 and re.search(r'\d', list_temp[-3]):
                # Remove last two words (unit) (e.g. "square meters")
                ans = " ".join(list_temp[:-2])

    # get rid of x=, y=
    if not in_the_form:
        ans = ans.split('=')[-1].strip()

    # get rid of , except in interval notation
    if not separated_by_commas and not re.search(r'\(\d+,\d+\)', ans):
        pattern = r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?'
        ans = re.sub(pattern, lambda x: x.group().replace(",", ""), ans)

    # get rid of %
    ans = ans.rstrip("%")

    # get rid of °
    ans = ans.rstrip("°")

    # split " \\in "
    ans = ans.split(" \\in ")[-1].strip()

    # Remove trailing zeros from decimals
    if re.match(r'^-?\d+\.\d+$', ans):
        ans = ans.rstrip('0').rstrip('.')

    return ans


def extract_answer(pred_str, in_the_form=False, separated_by_commas=False, exhaust=False):
    pred = []

    if 'boxed' in pred_str:
        pred = extract_boxed_answers(pred_str)
    elif ('answer is' in pred_str):
        temp = pred_str.split('answer is')[-1].strip()
        # remove the rest of the words after the answer
        temp = temp.split('.\n')[0].strip()
        temp = temp.split('. ')[0].strip()
        if " or " in temp:
            pred = temp.split(" or ")
        else:
            pred = [temp]

    else:
        # use the last word that has number
        words = pred_str.split()
        for word in reversed(words):
            if re.search(r'\d', word):
                pred.append(word)
                break
        # TODO: handle exceptional cases for labels without number
        if pred == []:
            pred.append(pred_str)

    # multiple line
    _pred = []
    for ans in pred:
        ans = ans.strip()
        ans = ans.lstrip(":")
        ans = ans.rstrip(".")
        ans = ans.rstrip("/")
        # remove $ signs
        ans = ans.replace("\\$", "")
        ans = ans.replace("$", "")

        # handle minor cases
        ans = handle_minor_cases(ans, in_the_form, separated_by_commas)
        ans = strip_string(ans)
        _pred.append(ans)

    if exhaust:
        return _pred
    else:
        return _pred[-1] if _pred else ""


def extract_mathcollege_answer(pred_str, in_the_form=False, separated_by_commas=False, exhaust=False):
    pred = []

    # Handle more LaTeX patterns
    if '\\begin{align}' in pred_str or '\\begin{equation}' in pred_str:
        # Extract from LaTeX environments
        pattern = r'\\begin\{(?:align|equation)\*?\}(.*?)\\end\{(?:align|equation)\*?\}'
        matches = re.findall(pattern, pred_str, re.DOTALL)
        if matches:
            for match in matches:
                # Extract the last line or expression after an = sign
                lines = match.strip().split('\\\\')
                last_line = lines[-1].strip()
                if '=' in last_line:
                    pred.append(last_line.split('=')[-1].strip())
                else:
                    pred.append(last_line)

    # Keep existing extraction methods
    if not pred and extract_solutionis_dollar_expression(pred_str):
        pred = extract_solutionis_dollar_expression(pred_str)
    elif not pred and 'boxed' in pred_str:
        pred = extract_boxed_answers(pred_str)
    elif not pred and ('answer is' in pred_str.lower() or 'solution is' in pred_str.lower()):
        # More flexible answer pattern matching
        pattern = r'(?:answer|solution)(?:\s+is|:)\s+(.*?)(?:$|\.|\n|,\s*(?:which|where|because|thus))'
        matches = re.findall(pattern, pred_str.lower(), re.IGNORECASE)
        if matches:
            temp = matches[-1].strip()
            if " or " in temp:
                pred = temp.split(" or ")
            else:
                pred = [temp]
    elif not pred and extract_last_dollar_expression(pred_str):
        pred = extract_last_dollar_expression(pred_str)

    # Handle cases where the answer is after keywords like "equals", "yields", "gives", etc.
    if not pred:
        pattern = r'(?:equals|yields|gives|results in|evaluates to)\s+(.*?)(?:$|\.|\n)'
        matches = re.findall(pattern, pred_str.lower(), re.IGNORECASE)
        if matches:
            pred = [matches[-1].strip()]

    # Fallback to numeric extraction
    if not pred:
        words = pred_str.split()
        for word in reversed(words):
            if re.search(r'(?:\d|\\pi|\\infty|infinity)', word):
                pred.append(word)
                break
        if not pred:
            pred.append(pred_str)

    # Process each candidate answer
    _pred = []
    for ans in pred:
        ans = ans.strip()
        ans = ans.lstrip(":")
        ans = ans.rstrip(".")
        ans = ans.rstrip("/")

        # Handle LaTeX forms
        ans = ans.replace("\\$", "")
        ans = ans.replace("$", "")

        # Normalize spaces in specific LaTeX constructs
        ans = re.sub(r'\\frac\s+{', r'\\frac{', ans)
        ans = re.sub(r'}\s+{', r'}{', ans)

        # Handle square roots with and without braces
        ans = re.sub(r'\\sqrt\s+(\w)', r'\\sqrt{\1}', ans)

        # Clean spaces only if fraction is not involved
        if 'frac' not in ans:
            ans = ans.replace(" ", "")
            ans = ans.replace("{", "")
            ans = ans.replace("}", "")

        # Handle minor cases and strip
        ans = handle_minor_cases(ans, in_the_form, separated_by_commas)
        ans = strip_string(ans)
        _pred.append(ans)

    if exhaust:
        return _pred
    else:
        return _pred[-1] if _pred else ""


def parse_gsm_answer(input_str):
    """Extract numerical answer from a string."""
    pattern = r"([0-9]*)"
    matches = re.findall(pattern, input_str)

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            return str(solution)

    return None


def parse_aime_answer(input_str):
    pattern = r"([0-9]*)"
    matches = re.findall(pattern, input_str)

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            return str(solution)

    return None


def extract_math_answer(reasoning, dataset):
    answer = []

    # Handle special dataset cases
    in_the_form = 'in the form' in dataset
    separated_by_commas = 'separated by commas' in dataset

    if separated_by_commas:
        # Existing comma-separated handling
        for ans in extract_answer(reasoning, in_the_form=in_the_form, separated_by_commas=True, exhaust=True):
            ans = ans.replace("(", "").replace(")", "").replace(
                "[", "").replace("]", "")
            temp = [a.strip() for a in ans.split(",")]
            answer.append(temp)

            # Handle fractions
            num_temp = []
            frac_flag = False
            for a in temp:
                if 'frac' in a:
                    frac_a = parse_fraction(a)
                    if frac_a is not None:
                        num_temp.append(frac_a)
                        frac_flag = True
                    else:
                        num_temp.append(a)
                else:
                    num_temp.append(a)
            if frac_flag:
                answer.append(num_temp)
        return answer[0]
    elif 'gsm8k' in dataset:
        return parse_gsm_answer(reasoning)
    elif 'aime2024' in dataset or 'aime2025' in dataset:
        return parse_aime_answer(reasoning)
    elif 'collegemath' in dataset:
        # Use the improved collegemath parser
        answers = extract_mathcollege_answer(
            reasoning, in_the_form=in_the_form, exhaust=True)
        for ans in answers:
            # Handle "and" in answers
            if regex.search(r"\\text\{\s*and\s*\}", ans):
                temp = [a.strip()
                        for a in regex.split(r"\\text\{\s*and\s*\}", ans)]
                for a in temp:
                    answer.append(a.strip())
                    if 'frac' in a:
                        frac_a = parse_fraction(a)
                        if frac_a is not None:
                            answer.append(frac_a)
            else:
                answer.append(ans.strip())
                # Try to normalize the expression to canonical form
                canonical = normalize_math_expression(ans.strip())
                if canonical != ans.strip():
                    answer.append(canonical)
                # Also add decimal version of fractions
                if 'frac' in ans:
                    frac_a = parse_fraction(ans)
                    if frac_a is not None:
                        answer.append(frac_a)

        try:
            return answer[0] if answer else None
        except:
            return None
    else:
        # General case (math500)
        answers = extract_answer(
            reasoning, in_the_form=in_the_form, exhaust=True)
        for ans in answers:
            # Handle LaTeX "and" in text
            if regex.search(r"\\text\{\s*and\s*\}", ans):
                temp = [a.strip()
                        for a in regex.split(r"\\text\{\s*and\s*\}", ans)]
                for a in temp:
                    answer.append(a.strip())
                    # Add normalized form
                    canonical = normalize_math_expression(a.strip())
                    if canonical != a.strip():
                        answer.append(canonical)
                    # Add decimal form if it's a fraction
                    if 'frac' in a:
                        frac_a = parse_fraction(a)
                        if frac_a is not None:
                            answer.append(frac_a)
            # Handle "and" in plain text
            elif " and " in ans.lower():
                temp = [a.strip() for a in ans.lower().split(" and ")]
                for a in temp:
                    answer.append(a.strip())
            else:
                answer.append(ans.strip())
                # Add normalized form
                canonical = normalize_math_expression(ans.strip())
                if canonical != ans.strip():
                    answer.append(canonical)
                # Add decimal form if it's a fraction
                if 'frac' in ans:
                    frac_a = parse_fraction(ans)
                    if frac_a is not None:
                        answer.append(frac_a)

        try:
            return answer[0] if answer else None
        except:
            return None


def latex_to_plain_fraction(expr):
    # \frac{a}{b} -> (a)/(b) 로 변환
    pattern = r'\\{1,2}frac{([^{}]+)}{([^{}]+)}'
    while re.search(pattern, expr):
        expr = re.sub(pattern, r'(\1)/(\2)', expr)
    return expr


def preprocess_expression(expr):
    expr_input = expr

    # frac{a}{b} → a/b
    expr = latex_to_plain_fraction(expr)

    expr = re.sub(r'\^\{([^{}]+)\}', r'^\1', expr)  # x^{2} → x^2
    expr = re.sub(r'sqrt\{([^{}]+)\}', r'sqrt(\1)', expr)  # sqrt{x} → sqrt(x)
    expr = re.sub(r'\\sqrt{([^{}]+)}', r'sqrt(\1)', expr)

    expr = expr.replace('\\sin', 'sin')
    expr = expr.replace('\\cos', 'cos')
    expr = expr.replace('\\tan', 'tan')
    expr = expr.replace('\\ln', 'ln')
    expr = expr.replace('\\log', 'log')
    expr = expr.replace('\\exp', 'exp')
    expr = expr.replace('\\sqrt', 'sqrt')
    # LaTeX functions to sympy-style: \sin{x} -> sin(x)
    func_list = ['sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt']
    for func in func_list:
        expr = re.sub(rf'\\?{func}\s*{{([^{{}}]+)}}', rf'{func}(\1)', expr)

        # \\sinx -> sin(x)
        expr = re.sub(rf'\\?{func}(\d+[a-zA-Z0-9_]+)', rf'{func}(\1)', expr)
        # \sin x -> sin(x) (공백 있는 경우)
        expr = re.sub(rf'\\?{func}\s+([a-zA-Z0-9_]+)', rf'{func}(\1)', expr)
        # )func ->)*func
        expr = re.sub(rf'\)({func})(?=\()', rf')*{func}', expr)

    # x^{2} -> x**(2)
    expr = re.sub(r'([a-zA-Z0-9_]+)\^\{([^{}]+)\}', r'\1**(\2)', expr)
    expr = re.sub(r'([a-zA-Z0-9_]+)\^([a-zA-Z0-9_]+)',
                  r'\1**\2', expr)  # x^2 -> x**2 (without braces)
    expr = expr.replace('^', '**')

    # )x → )*x
    # expr = re.sub(r'\)(?=[a-zA-Z_])', r')*', expr)

    # 곱셈 생략된 곳 처리: 4x → 4*x, x2 → x*2
    expr = re.sub(r'(\d)([a-zA-Z_])', r'\1*\2', expr)  # 2x → 2*x
    expr = re.sub(r'(\d)([a-zA-Z_]\w*)', r'\1*\2', expr)  # 2xyz → 2*x*y*z

    print(f"before process : {expr_input}, after : {expr}")
    return expr


def compare_answers(parsed_label, parsed_answer):
    # If they're identical strings, return True immediately
    if parsed_label == parsed_answer:
        return True

    try:
        # Handle common equivalence cases
        if _is_numeric_equivalent(parsed_label, parsed_answer):
            return True

        # Clean expressions before symbolic comparison
        expr1 = re.sub(r'(\d)([a-zA-Z_]\w*)', r'\1*\2', str(parsed_label))
        expr2 = re.sub(r'(\d)([a-zA-Z_]\w*)', r'\1*\2', str(parsed_answer))

        # Collect all variables from both expressions
        variables = set(re.findall(
            r'[a-zA-Z_]\w*', str(parsed_label) + ' ' + str(parsed_answer)))
        variables = {v for v in variables if v not in [
            'sin', 'cos', 'tan', 'log', 'exp', 'sqrt']}

        # Create sympy symbols for each variable
        symbols = {var: sp.Symbol(var) for var in variables}

        # Parse the expressions with better error handling
        try:
            expr1 = sp.sympify(expr1, locals=symbols)
            expr2 = sp.sympify(expr2, locals=symbols)
            return sp.simplify(expr1 - expr2) == 0
        except:
            # Try parsing as rational numbers if possible
            try:
                expr1 = sp.Rational(expr1)
                expr2 = sp.Rational(expr2)
                return expr1 == expr2
            except:
                pass

        # Try numerical evaluation for specific values
        return _numerical_equivalence_test(expr1, expr2, symbols)
    except Exception as e:
        # Fall back to string comparison if symbolic methods fail
        return _normalize_string(parsed_label) == _normalize_string(parsed_answer)


def _is_numeric_equivalent(val1, val2):
    """Check if two values are numerically equivalent"""
    try:
        # Try converting to float and comparing
        num1 = float(val1)
        num2 = float(val2)
        return abs(num1 - num2) < 1e-10
    except:
        return False


def _normalize_string(s):
    """Normalize string representation for comparison"""
    s = str(s).lower().strip()
    # Remove spaces, normalize fractions, etc.
    s = re.sub(r'\s+', '', s)
    s = s.replace('\\frac', 'frac')
    return s


def _numerical_equivalence_test(expr1, expr2, symbols):
    """Test equivalence by substituting random values"""
    import random
    try:
        # Try multiple random values to reduce false positives
        for _ in range(3):
            values = {sym: random.uniform(-10, 10) for sym in symbols}
            val1 = float(expr1.subs(values))
            val2 = float(expr2.subs(values))
            if abs(val1 - val2) > 1e-10:
                return False
        return True
    except:
        return False


def normalize_math_expression(expr):
    """Normalize mathematical expressions to canonical form"""
    try:
        # Clean and prepare expression
        expr = strip_string(expr)

        # Handle common mathematical constants
        expr = expr.replace('\\pi', 'pi')
        expr = expr.replace('\\infty', 'oo')  # sympy uses 'oo' for infinity

        # Convert expression to sympy form
        expr = re.sub(r'(\d)([a-zA-Z_]\w*)', r'\1*\2', expr)

        # Extract variables
        variables = set(re.findall(r'[a-zA-Z_]\w*', expr))
        variables = {v for v in variables if v not in [
            'sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'pi', 'oo']}

        # Create symbols and try to parse
        symbols = {var: sp.Symbol(var) for var in variables}
        locals_dict = symbols.copy()
        locals_dict.update({'pi': sp.pi, 'oo': sp.oo})

        # Convert to canonical form
        sympy_expr = sp.sympify(expr, locals=locals_dict)
        return str(sympy_expr)
    except:
        # If conversion fails, return the original cleaned string
        return expr

#####################################
# for majority voting
#####################################


def normalize_expression(expr):
    try:
        expr = preprocess_expression(expr)

        variables = set(re.findall(r'[a-zA-Z_]\w*', expr))
        symbols = sp.symbols(' '.join(variables))
        expr = sp.sympify(expr)
        simplified = sp.simplify(expr)
        print(f"expr: {expr}, simplified: {simplified}")
        return str(simplified)
    except Exception:
        return expr


def most_frequent(answers):
    """Return the most frequent original answer based on normalized expressions."""
    answers = [a for a in answers if a is not None]
    if not answers:
        return None

    normalized_map = {}

    for ans in answers:
        norm = normalize_expression(ans)
        if norm not in normalized_map:
            normalized_map[norm] = []
        normalized_map[norm].append(ans)

    # Count normalized expressions
    normalized_counts = {k: len(v) for k, v in normalized_map.items()}
    most_common_norm = max(normalized_counts.items(), key=lambda x: x[1])[0]
    print("original_answers: ", answers)
    print("most_common_norm: ", most_common_norm)
    print("normalized_map: ", normalized_map)
    print("returned: ", normalized_map[most_common_norm][0])
    # Return one of the original answers (first) that matched the most common normalized form
    return normalized_map[most_common_norm][0]
