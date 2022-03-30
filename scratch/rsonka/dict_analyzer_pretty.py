"""
Rita Sonka 3/30/2022
Made to create reference illustrations of the structure of .npy files 
and other large dictionaries. 
"""

#234567890123456789012345678901234567890123456789012345678901234567890123456789
def dict_analyzer_pretty(d, max_line=80, ignore_type=None, ignore_after=1, 
                         do_padding=True,s_key="", starting_space=""):
    """ Trawls through a large nested dictionary and creates a string to 
    illustrate its structure, reporting each key, the type() of that key's
    value, and a preview of the string representation of that key's value;
    except, it ignores excess (> ignore_after) keys of ignore_type on
    a given dictionary level. 
    ---- Args -------
    d               : dict : the dictionary to traverse
    max_line        : int  : line character # to end value previews at
    ignore_type     : type : Ignore excess keys of this type @ one dict level
                           : NOTE: may be "numpy.int64" or similar!
    ignore_after    : int  : # of keys of ignore type to show (on a given 
                           : dict level) before ignoring
    do_padding      : bool : Pad to align colons like in this docstring?
    s_key           : ??   : "start key," recursive parameter.
    starting_space  : str  : recursive parameter.
    ---- Returns ----
    A string that illustrates the dictionary,  of the following form,
    with all colons on a given level aligned via space padding if
    do_padding=True (assume  below that keys don't have dictionary
    values unless specified):
    
    {
        key1 : type(key1) <preview of key1>
        key2 : type(key2) <preview of key2>
        key3 : dict       {
            subkey1 : type(subkey1) <preview of subkey1>
            subkey2 : type(subkey2) <preview of subkey2>
            subkey3 : dict          {
                subsubkey1 : type(subsubkey1) <preview of subsubkey1>
            } #(subkey3)
        } #(key3) 
    } #()
    
    except if you set the ignore settings and it ignores keys, it will report
    how many were ignored in a given dictionary in parentheses after the 
    #(dictkeyname) following that dictionary preview's closing bracket.
    """
    ss = starting_space + "    "
    string = " {\n" #str(start_key) +
    max_key_len    = max([len(str(key)) for key in d.keys()])
    max_type_len   = max([len(str(type(d[key]))[8:-2]) for key in d.keys()])
    base_line_len  = len(ss) + max_key_len + 1 + max_type_len + 2
    value_space    = max(max_line - base_line_len, 3)
    seen_ignores   = 0
    key_space, type_space = " ", " " # for if do_padding turned off
    for key in d.keys():
        if type(key) == ignore_type:
            seen_ignores += 1
            if seen_ignores > ignore_after:
                continue
        if do_padding:
            key_space  = " " * (max_key_len  - len(str(key)))
            type_space = " " * (max_type_len - len(str(type(d[key]))[8:-2]))
        if type(d[key]) == dict:
            # f strings were being annoying about this for some reason.
            string = string + ss + str(key)  + key_space  + ":"  + \
                     str(type(d[key]))[8:-2] + type_space + " "  + \
                     dict_analyzer_pretty(d[key], s_key=key, starting_space=ss, 
                                          max_line=max_line,
                                         ignore_type=ignore_type,
                                         ignore_after=ignore_after) + "\n"
        else:
            if len(str(d[key])) <= value_space:
                va = d[key]
            else:
                va = str(d[key])[:value_space-3] + "..." #str(len(str(d[key])))
            string = string + ss + str(key)  + key_space  + ":"  + \
                     str(type(d[key]))[8:-2] + type_space + " "  + \
                     str(va) + "\n" 
    string = string + starting_space + "} #("+ str(s_key) + ")"
    if seen_ignores > 0:
        string = string + " (ignored " + str(seen_ignores) + " " + \
                 str(ignore_type)[8:-2] + ")"
    return string




