/*
Copyright (c) 2014, Justin Willmert
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the School of Physics and Astronomy, University of
      Minnesota nor the names of its contributors may be used to endorse or
      promote products derived from this software without specific prior
      written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL JUSTIN WILLMERT BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
 * File: pager.js
 * Author: Justin Willmert ⓒ 2014
 *
 * This library provides a simple interface which eases the creation of figure
 * pagers used in my research summaries.
 *
 * USAGE
 * =====
 *
 * BASIC USAGE
 *     The entry point for creating a pager is the Pager.link function. It is
 *     used to register an image with the pager system, and in the process, a
 *     panel of pager options is generated alongside the image.
 *
 * MULTIPLE NAMESPACES
 *
 * CSS STYLING
 *     The goal of the pager is to be descriptive enough that all elements are
 *     stylable so that the pager can be made to fit in with any page style.
 *
 *     The following sample CSS will set basic styles on the pager so that
 *     active elements are actively highlighted (with a.pager and
 *     a.pager.active), the option labels are shown in bold and right-aligned
 *     within the table cell (.pager.label), and most importantly, the pager
 *     options and figure image (.pager.container and img.pager, respectively)
 *     are shown side-by-side by ensuring that the pager container and the
 *     image are treated as inline-block elements.
 *
 *     <style type="text/css">
 *         /* Whole pager option table * /
 *         .pager.container {
 *             display: inline-block;
 *         }
 *         /* Associated pager image * /
 *         img.pager {
 *             display: inline-block;
 *             vertical-align: middle;
 *         }
 *
 *         /* Option group labels * /
 *         .pager.label {
 *             text-align: right;
 *             font-weight: bold;
 *         }
 *
 *         /* Generic styles for all options * /
 *         a.pager {
 *             padding: 2px 4px;
 *         }
 *         /* Styling for the active options * /
 *         a.pager.active {
 *             background-color: #c0c0c0;
 *             border-radius: 4px;
 *         }
 *     </style>
 */

/*****************************************************************************
 *             POLYFILLS FROM THE MOZILLA DEVELOPER NETWORK                  *
 *****************************************************************************/

// From https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Function/bind
if (!Function.prototype.bind) {
  Function.prototype.bind = function (oThis) {
    if (typeof this !== "function") {
      // closest thing possible to the ECMAScript 5
      // internal IsCallable function
      throw new TypeError("Function.prototype.bind - what is trying to be bound is not callable");
    }

    var aArgs = Array.prototype.slice.call(arguments, 1),
        fToBind = this,
        fNOP = function () {},
        fBound = function () {
          return fToBind.apply(this instanceof fNOP && oThis
                 ? this
                 : oThis,
                 aArgs.concat(Array.prototype.slice.call(arguments)));
        };

    fNOP.prototype = this.prototype;
    fBound.prototype = new fNOP();

    return fBound;
  };
}

// From https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object/keys
if (!Object.keys) {
  Object.keys = (function () {
    'use strict';
    var hasOwnProperty = Object.prototype.hasOwnProperty,
        hasDontEnumBug = !({toString: null}).propertyIsEnumerable('toString'),
        dontEnums = [
          'toString',
          'toLocaleString',
          'valueOf',
          'hasOwnProperty',
          'isPrototypeOf',
          'propertyIsEnumerable',
          'constructor'
        ],
        dontEnumsLength = dontEnums.length;

    return function (obj) {
      if (typeof obj !== 'object' && (typeof obj !== 'function' || obj === null)) {
        throw new TypeError('Object.keys called on non-object');
      }

      var result = [], prop, i;

      for (prop in obj) {
        if (hasOwnProperty.call(obj, prop)) {
          result.push(prop);
        }
      }

      if (hasDontEnumBug) {
        for (i = 0; i < dontEnumsLength; i++) {
          if (hasOwnProperty.call(obj, dontEnums[i])) {
            result.push(dontEnums[i]);
          }
        }
      }
      return result;
    };
  }());
}

// From https://developer.mozilla.org/en-US/docs/Web/API/EventTarget.addEventListener
(function() {
  if (!Event.prototype.preventDefault) {
    Event.prototype.preventDefault=function() {
      this.returnValue=false;
    };
  }
  if (!Event.prototype.stopPropagation) {
    Event.prototype.stopPropagation=function() {
      this.cancelBubble=true;
    };
  }
  if (!Element.prototype.addEventListener) {
    var eventListeners=[];

    var addEventListener=function(type,listener /*, useCapture (will be ignored) */) {
      var self=this;
      var wrapper=function(e) {
        e.target=e.srcElement;
        e.currentTarget=self;
        if (listener.handleEvent) {
          listener.handleEvent(e);
        } else {
          listener.call(self,e);
        }
      };
      if (type=="DOMContentLoaded") {
        var wrapper2=function(e) {
          if (document.readyState=="complete") {
            wrapper(e);
          }
        };
        document.attachEvent("onreadystatechange",wrapper2);
        eventListeners.push({object:this,type:type,listener:listener,wrapper:wrapper2});

        if (document.readyState=="complete") {
          var e=new Event();
          e.srcElement=window;
          wrapper2(e);
        }
      } else {
        this.attachEvent("on"+type,wrapper);
        eventListeners.push({object:this,type:type,listener:listener,wrapper:wrapper});
      }
    };
    var removeEventListener=function(type,listener /*, useCapture (will be ignored) */) {
      var counter=0;
      while (counter<eventListeners.length) {
        var eventListener=eventListeners[counter];
        if (eventListener.object==this && eventListener.type==type && eventListener.listener==listener) {
          if (type=="DOMContentLoaded") {
            this.detachEvent("onreadystatechange",eventListener.wrapper);
          } else {
            this.detachEvent("on"+type,eventListener.wrapper);
          }
          break;
        }
        ++counter;
      }
    };
    Element.prototype.addEventListener=addEventListener;
    Element.prototype.removeEventListener=removeEventListener;
    if (HTMLDocument) {
      HTMLDocument.prototype.addEventListener=addEventListener;
      HTMLDocument.prototype.removeEventListener=removeEventListener;
    }
    if (Window) {
      Window.prototype.addEventListener=addEventListener;
      Window.prototype.removeEventListener=removeEventListener;
    }
  }
})();

// From https://developer.mozilla.org/en-US/docs/Web/API/Element.classList
/*
 * classList.js: Cross-browser full element.classList implementation.
 * 2014-01-31
 *
 * By Eli Grey, http://eligrey.com
 * Public Domain.
 * NO WARRANTY EXPRESSED OR IMPLIED. USE AT YOUR OWN RISK.
 */
/*! @source http://purl.eligrey.com/github/classList.js/blob/master/classList.js*/

if ("document" in self && !("classList" in document.createElement("_"))) {

    (function (view) {
        "use strict";
        if (!('Element' in view)) return;

        var
              classListProp = "classList"
            , protoProp = "prototype"
            , elemCtrProto = view.Element[protoProp]
            , objCtr = Object
            , strTrim = String[protoProp].trim || function () {
                return this.replace(/^\s+|\s+$/g, "");
            }
            , arrIndexOf = Array[protoProp].indexOf || function (item) {
                var
                      i = 0
                    , len = this.length
                ;
                for (; i < len; i++) {
                    if (i in this && this[i] === item) {
                        return i;
                    }
                }
                return -1;
            }
            // Vendors: please allow content code to instantiate DOMExceptions
            , DOMEx = function (type, message) {
                this.name = type;
                this.code = DOMException[type];
                this.message = message;
            }
            , checkTokenAndGetIndex = function (classList, token) {
                if (token === "") {
                    throw new DOMEx(
                          "SYNTAX_ERR"
                        , "An invalid or illegal string was specified"
                    );
                }
                if (/\s/.test(token)) {
                    throw new DOMEx(
                          "INVALID_CHARACTER_ERR"
                        , "String contains an invalid character"
                    );
                }
                return arrIndexOf.call(classList, token);
            }
            , ClassList = function (elem) {
                var
                      trimmedClasses = strTrim.call(elem.getAttribute("class") || "")
                    , classes = trimmedClasses ? trimmedClasses.split(/\s+/) : []
                    , i = 0
                    , len = classes.length
                ;
                for (; i < len; i++) {
                    this.push(classes[i]);
                }
                this._updateClassName = function () {
                    elem.setAttribute("class", this.toString());
                };
            }
            , classListProto = ClassList[protoProp] = []
            , classListGetter = function () {
                return new ClassList(this);
            }
        ;
        // Most DOMException implementations don't allow calling DOMException's toString()
        // on non-DOMExceptions. Error's toString() is sufficient here.
        DOMEx[protoProp] = Error[protoProp];
        classListProto.item = function (i) {
            return this[i] || null;
        };
        classListProto.contains = function (token) {
            token += "";
            return checkTokenAndGetIndex(this, token) !== -1;
        };
        classListProto.add = function () {
            var
                  tokens = arguments
                , i = 0
                , l = tokens.length
                , token
                , updated = false
            ;
            do {
                token = tokens[i] + "";
                if (checkTokenAndGetIndex(this, token) === -1) {
                    this.push(token);
                    updated = true;
                }
            }
            while (++i < l);

            if (updated) {
                this._updateClassName();
            }
        };
        classListProto.remove = function () {
            var
                  tokens = arguments
                , i = 0
                , l = tokens.length
                , token
                , updated = false
            ;
            do {
                token = tokens[i] + "";
                var index = checkTokenAndGetIndex(this, token);
                if (index !== -1) {
                    this.splice(index, 1);
                    updated = true;
                }
            }
            while (++i < l);

            if (updated) {
                this._updateClassName();
            }
        };
        classListProto.toggle = function (token, force) {
            token += "";

            var
                  result = this.contains(token)
                , method = result ?
                    force !== true && "remove"
                :
                    force !== false && "add"
            ;

            if (method) {
                this[method](token);
            }

            return !result;
        };
        classListProto.toString = function () {
            return this.join(" ");
        };

        if (objCtr.defineProperty) {
            var classListPropDesc = {
                  get: classListGetter
                , enumerable: true
                , configurable: true
            };
            try {
                objCtr.defineProperty(elemCtrProto, classListProp, classListPropDesc);
            } catch (ex) { // IE 8 doesn't support enumerable:true
                if (ex.number === -0x7FF5EC54) {
                    classListPropDesc.enumerable = false;
                    objCtr.defineProperty(elemCtrProto, classListProp, classListPropDesc);
                }
            }
        } else if (objCtr[protoProp].__defineGetter__) {
            elemCtrProto.__defineGetter__(classListProp, classListGetter);
        }
    }(self));
}

/*****************************************************************************
 *                                 PAGER                                     *
 *****************************************************************************/

var Pager = function(namespace) {
    this.namespace = (typeof namespace!=='undefined' ? namespace : 'pager');
    this.counter = 0;
    this.params = {};
    this.registry = {};

    var reset = function() {
        var hash  = document.location.hash;
        if (Object.keys(this.registry).indexOf(hash) == -1) {
            return;
        }
        this.deserialize(document.location.search.substring(1));
    }
    document.addEventListener('DOMContentLoaded', reset.bind(this), false);
}

Pager.prototype.serialize = function() {
    var query = [];
    var params = this.params;
    for (var p in params) {
        if (!params.hasOwnProperty(p))
            continue;
        query.push(encodeURIComponent(p) + "=" + encodeURIComponent(params[p]));
    }
    return query.join("&");
}

Pager.prototype.deserialize = function(query) {
    /* Modeled on https://stackoverflow.com/a/2880929 */
    var decode = function (s) { return decodeURIComponent(s.replace(/\+/g, " ")); };
    var search = /([^&=]+)=?([^&]*)/g;
    var params = {};
    while (match = search.exec(query)) {
        params[decode(match[1])] = decode(match[2]);
    }
    this.setparams(params);
    return params;
}

/**
 * Pager.link(img_sel, options, cb_gen)
 *
 * Registers an <img> element with the pager. Changing option values will
 * have no effect unless the image element and an appropriate callback
 * function have been registered with the pager system. This also inserts
 * the option selector links into the DOM as the previous sibling to the
 * image selected.
 *
 * INPUTS
 *     img_sel    A querySelector-style selector string which is used to
 *                select the figure which is to be updated.
 *
 *     options    An associative array which defines all the options and
 *                parameters that the pager will make use of. The option group
 *                is defined by a string in one of the two following forms:
 *
 *                  1. If no pipe symbol is found, then the string is used
 *                     as-is for display and internal use.
 *
 *                  2. If a pipe-symbol is found, then the first half is
 *                     the human-friendly string, while the second half is
 *                     the string used progmatically.
 *
 *                The possible options for the group must be given in one of
 *                the following recognized input forms:
 *
 *                  => An associative string array in the same format as the
 *                     options array, with the added special case that a
 *                     empty string is translated to a manual line-break
 *                     <br> for formatting reasons.
 *
 *                  => A Pager.DataList object.
 *
 *     cb_gen     A callback function which takes in as an input an
 *                associative array of the current option selections. It
 *                should then return a string which is the path name to the
 *                image which should be shown.
 *
 * EXAMPLE
 *
 *     opts = { 'Year|yr': ['2013','2014'] };
 *     pager.link('#yearly_plot', opts, function(p){ return p['yr']+'.png'; });
 *
 */
Pager.prototype.link = function(img_sel, options, cb_gen) {
    // Make sure the image selection actually works.
    var img = document.querySelector(img_sel);
    if (img === null)
    {
        console.warn("[pager.link]: '" + img_sel
            + "' matches no element. Not registering.");
        return;
    }

    // Ensure the callback function make sense to call
    if (typeof cb_gen === "function")
    {
        // Simply save the function pointer
        this.registry[img_sel] = cb_gen;
    }
    else
    {
        console.warn("[pager.link]: Invalid function. Not registering '"
            + img_sel + '".');
        return;
    }

    // Claim the image as a pager image
    img.classList.add('pager');

    // Now do the hard work of building the option panels

    // The pager will consist of the options which can be selected for
    // a given image. We'll assume that the user is providing the image
    // (which is how we identify a pager anyway, by the selector string),
    // so all we care about is the collection of links which must be built.
    var _ = document.createElement('table');

    // Basic table properties.
    _.classList.add('pager','container');
    if (img.id !== "")
        _.id = img.id + '_options';
    else
    {
        console.warn('[pager.link]: All pager images should have an ID!');
        _.id = this.namespace + this.counter;
    }
    this.counter++;

    // Generate permalink cell and link:
    var permalink = document.createElement('a');
    permalink.href = "#";
    permalink.id = img.id + '_permalink';
    permalink.appendChild(document.createTextNode('permalink'));
    permalink.addEventListener('click', function(evt) {
            evt.preventDefault();
            return void(0);
        }, false);
    permalink.classList.add('pager', 'permalink');

    var permacell = document.createElement('td');
    permacell.setAttribute('colspan', 2);
    permacell.classList.add('pager', 'permalink');
    permacell.appendChild(permalink);

    var permarow = document.createElement('tr');
    permarow.classList.add('pager', 'permalink');
    permarow.appendChild(permacell);
    _.appendChild(permarow);

    // Split up an element around the '|' if necessary, otherwise duplicate
    // into two values.
    var splitpipe = function(str)
    {
        var pos = str.indexOf('|');
        if (pos == -1)
            return [str,str];
        else
        {
            var human = str.substr(0,pos);
            var machine = str.substr(pos+1);
            return [human,machine];
        }
    };

    var keys = Object.keys(options);
    for (var ii=0; ii<keys.length; ++ii)
    {
        // Deconstruct the human-machine parts
        var key = splitpipe(keys[ii]);
        var hkey = key[0];
        var mkey = key[1];

        // New row to identify the option group
        var r = document.createElement('tr');
        var c1 = document.createElement('td');
        c1.classList.add('pager','label');
        c1.appendChild(document.createTextNode(hkey));
        r.appendChild(c1);
        _.appendChild(r);

        var c2 = document.createElement('td');
        c2.classList.add('pager','options');
        r.appendChild(c2);

        // Building the remainder of the UI depends on what the input type
        // of the argument was.
        var value = options[keys[ii]];
        // For a Pager.DataList object which scrolls through a list:
        if (value instanceof Pager.DataList)
        {
            // Insert the DataList into the params object which will save it
            // for later.
            this.params[mkey] = value;

            // Buttons and inputs need to be attached to a form
            var form = document.createElement('form');
            form.classList.add('pager');
            form.addEventListener('submit', function(evt) {
                evt.preventDefault();
                return void(0);
            }, false);

            // Identify the selectors by the form
            form.setAttribute('namespace', this.namespace);
            form.setAttribute('paramkey', mkey);

            // Create the two arrows and the input
            var rewind    = document.createElement('input');
            rewind.classList.add('pager');
            rewind.type   = 'button';
            rewind.value  = '≪';
            var advance   = document.createElement('input');
            advance.classList.add('pager');
            advance.type  = 'button';
            advance.value = '≫';

            var input     = document.createElement('input');
            input.classList.add('pager');
            input.type    = 'text';
            input.size    = 6;

            // Attach the buttons to events which will do the right thing
            rewind.addEventListener('click', value.rewind.bind(value), false);
            rewind.addEventListener('click',
                this.setopt.bind(this, mkey, undefined), false);

            advance.addEventListener('click', value.advance.bind(value),
                false);
            advance.addEventListener('click',
                this.setopt.bind(this, mkey, undefined), false);

            // Also let the user input a value and then skip to the
            // corresponding figure.
            input.addEventListener('change', this.setopt.bind(this, mkey),
                false);

            // Insert them into the form in order.
            form.appendChild(rewind);
            form.appendChild(input);
            form.appendChild(advance);

            c2.appendChild(form);
        }
        // The default case just assumes the value is just an array of strings
        // to use with links.
        else if (value instanceof Array)
        {
            // Insert each individual option as a link
            for (var jj=0; jj<value.length; ++jj)
            {
                var val = splitpipe(value[jj]);
                var hval = val[0];
                var mval = val[1];

                // Choose the first element as the default
                if (jj == 0)
                    this.params[mkey] = mval;

                // For the special case that the key and value are empty,
                // use this as a manual line-breaking mechanism.
                if (hval == '' && mval == '')
                {
                    var a = document.createElement('br');
                    // Also annotate to know that this is an explicit line
                    // break that shouldn't be removed in a gridalign()
                    // reflow.
                    a.classList.add('pager_explicit');
                }
                // Otherwise construct a useful link
                else
                {
                    var a = document.createElement('a');
                    a.href = "javascript:void(0);";
                    a.addEventListener('click', this.setopt.bind(this, mkey,
                          mval, true), false);
                    a.appendChild(document.createTextNode(hval));
                    a.classList.add('pager');

                    // Make finding all relevant links easy
                    a.setAttribute('namespace', this.namespace);
                    a.setAttribute('paramkey', mkey);
                    a.setAttribute('paramval', mval);

                }
                // Insert the link with a space otherwise everything squishes
                // together if no extra CSS styles are included.
                c2.appendChild(a);
                c2.appendChild(document.createTextNode(" "));
            }
        }
        else
        {
            console.error('Unrecognized pager group value');
        }
    }

    // Attach the finished options panel into the main document DOM right
    // before the relevant <img>.
    img.parentNode.insertBefore(_, img);
}

/**
 * Pager.setparams(params)
 *
 * Sets all parameters to a value by reading the key-val pairs from params.
 * This is useful for setting the default parameters when the page is
 * initially loaded.
 *
 * INPUTS
 *     params    An associative array or object of key-value pairs which
 *               will determine the recognized parameter values.
 */
Pager.prototype.setparams = function(params) {
    var keys = Object.keys(params);
    for (var ii=0; ii<keys.length; ++ii)
        this.setopt(keys[ii], params[keys[ii]], ii==keys.length-1);
}

/**
 * Pager.setopt(option, value)
 *
 * Sets the parameter 'option' to the value 'value'. Setting an option
 * causes all registered pages to be updated.
 *
 * INPUTS
 *     option    The name of a particular option and should be a string
 *               since it is also used as the name of the key in an
 *               associative array.
 *
 *     value     Valid values depend on the type of data:
 *
 *                 1. If the option corresponds to a DataList, then the value
 *                    is taken from a variety of sources:
 *
 *                      a. If value is a string, then a new elements is
 *                         selected by invoking skipto(fromString()).
 *
 *                      b. If value is an Event object, then it is assumed that
 *                         the event was raised by the input text box and that
 *                         the new value is to be taken from its value.
 *
 *                      c. If value is undefined, then the current value stored
 *                         within the DataList is used.
 *
 *                 2. For typical enumerated string options, the value should
 *                    be a valid string.
 *
 *     update    Defaults to true where the figure images are updated. If
 *               false, then the stored parameters are update, but pager image
 *               sources are not updated. (This is most useful for setparams()
 *               which knows when the last parameter has been set.)
 */
Pager.prototype.setopt = function(option, value, update) {

    // Default to true if the value was not given
    update = (typeof update !== 'undefined' ? update : true);

    // DataList is updated using it's own API
    if (this.params[option] instanceof Pager.DataList)
    {
        // Only do an update if a value is provided
        if (typeof value !== 'undefined')
        {
            var d = this.params[option];

            // For a string input, convert and skip to
            if (typeof value == "string")
                d.skipto(d.fromString(value));
            // For an event, read the value from the input box
            else if (value instanceof Event)
                d.skipto(d.fromString(value.target.value));
        }

        // Now update what the display box will show
        var selstr = 'form[namespace="' + this.namespace + '"]'
                       + '[paramkey="' + option + '"]'
                       + ' > input[type="text"]';
        var input = document.querySelector(selstr);
        input.value = this.params[option].toString();
    }
    // Other arguments are just set directly.
    else
    {
        this.params[option] = value;

        // Go through the work of updating the class styling for all links so
        // that we can style them nicely in CSS.

        // First just set everything in this group to inactive
        var selstr = 'a[namespace="' + this.namespace + '"]'
                       + '[paramkey="' + option + '"]';
        var wholegroup = document.querySelectorAll(selstr);
        for (var ii=0; ii<wholegroup.length; ++ii)
        {
            wholegroup[ii].classList.remove('active');
            wholegroup[ii].classList.add('inactive');
        }

        // Now reset the current active option to be highlighted in all cases
        selstr += '[paramval="' + value + '"]';
        var active = document.querySelectorAll(selstr);
        for (var ii=0; ii<active.length; ++ii)
        {
            active[ii].classList.remove('inactive');
            active[ii].classList.add('active');
        }
    }

    if (update)
    {
        // Update all registered images last since this causes the biggest delay
        // as images are loaded across the network. Since this is a synchronous
        // process, make sure that all other UI changes are made first.
        var images = Object.keys(this.registry);
        for (var ii=0; ii<images.length; ++ii)
        {
            var imgsel = images[ii];
            var linksel = images[ii] + '_permalink';

            // Get the image source name from the registered callback
            var fn     = this.registry[imgsel];
            var imgsrc = fn(this.params);
            // Also get a handle to the image
            var img    = document.querySelector(imgsel);
            img.src    = imgsrc;
            // Then update the permalink
            var link   = document.querySelector(linksel);
            link.href  = '?' + this.serialize() + imgsel;
        }
    }
}

/**
 * Modifies button sizes and inserts line breaks so that the option link
 * buttons are aligned to a grid of a given width.
 *
 * Note that any explicit line breaks specified in the option list during
 * setup are preserved.
 *
 * INPUTS
 *     img_sel    A querySelector-style selector string which is used to
 *                select the figure which is to be updated.
 *
 *     nwide      An integer specifying how many columns across should be
 *                constructed.
 */
Pager.prototype.gridalign = function(img_sel, nwide) {
    // Verify identity as in Pager.link()
    var img = document.querySelector(img_sel);
    if (img === null)
    {
        //console.debug("[pager.gridalign]: '" + img_sel
        //    + "' matches no element.");
        return;
    }

    // Everything for this selector is registered by the table named by
    // img.id+'_options'.
    var tblname = img.id + '_options';

    // Remove implicit line breaks created during any prior call to
    // gridalign(). Note that we will explicitly ignore deleting the
    // explict line breaks setup by the user.
    var breaks = document.querySelectorAll('#'+tblname + ' br');

    for (var ii=0; ii<breaks.length; ++ii)
    {
        if (breaks[ii].classList.contains("pager_explicit"))
        {
            continue;
        }

        // If not an explicit break, remove it.
        var pp = breaks[ii].parentNode.removeChild(breaks[ii]);
    }

    // Make sure the container td element for the links does not have a
    // maximum width which will interfere.
    var opttd = document.querySelectorAll('#'+tblname + ' td.pager.options');
    for (var ii=0; ii<opttd.length; ++ii)
    {
        opttd[ii].setAttribute("style", "max-width:none;");
    }

    // Now enumerate all link elements:
    var links = document.querySelectorAll('#'+tblname + ' a');
    // First, reset styles to auto to get the natural width:
    for (var ii=0; ii<links.length; ++ii)
    {
        links[ii].style.width = 'auto';
    }
    // Now we can get the width of elements as rendered.
    maxwd = 0;
    for (var ii=0; ii<links.length; ++ii)
    {
        var rect = links[ii].getBoundingClientRect();
        // Unfortunately, rect.width isn't always available...
        var elwd = rect.right - rect.left;
        if (maxwd < elwd) {
            maxwd = elwd;
            //console.debug('[pager.gridalign]: maxwidth = ' + maxwd);
        }
    }

    // Now setup the options to match a grid. Do this td-by-td.
    for (var ii=0; ii<opttd.length; ++ii)
    {
        var els = opttd[ii].childNodes;
        var inrow = 0;
        for (var jj=0; jj<els.length; ++jj)
        {
            if (els[jj].nodeName == "BR")
            {
                // We've already deleted implicit breaks, so respect this and
                // reset the counter.
                inrow = 0;
                continue;
            }
            else if (els[jj].nodeName == "A")
            {
                // Increment counter
                inrow++;
                // Also now specify the size of the element, computed earlier
                els[jj].style.width = maxwd + 'px';
            }
            else
            {
                // Skip over unrecognized nodes. This should primarily just
                // be text nodes used to space the elements
                continue;
            }

            // Insert a break if we need to
            if (inrow >= nwide)
            {
                //console.debug('[pager.gridalign]: inserting break');
                var br = document.createElement('br');
                opttd[ii].insertBefore(br, els[jj+1]);
                inrow = 0;
            }
        }
    }
}

/**
 * Wraps the target image in a wrapper node, useful for applying extra
 * styling.
 *
 * The function should be called *after* Pager.link() in order for the pager
 * option table to not be wrapped as well.
 *
 * INPUTS
 *     img_sel    A querySelector-style selector string which is used to
 *                select the figure which is to be updated.
 *
 *     eltype     The element type of the wrapper to generate.
 */
Pager.prototype.wrap = function(img_sel, eltype) {
    var img = document.querySelector(img_sel);
    if (img === null) {
        console.warn("[pager.wrap]: '" + img_sel + "' matches no element.");
        return;
    }

    // Generate the new wrapper node
    var wrap = document.createElement(eltype);
    wrap.classList.add('pager', 'wrapper');
    wrap.id = img.id + '_wrapper';

    // Insert the wrapper node relative to the next sibling of the image.
    // Doing insertion with respect to the image itself causes problems once
    // we remove the image from the DOM.
    var parent = img.parentNode;
    parent.insertBefore(wrap, img.nextElementSibling);

    // Then remove img from the parent and re-parent it as a child of wrap.
    parent.removeChild(img);
    wrap.appendChild(img);
}

/**
 * A Pager.DataList is a valid value type for Pager.link which creates an input
 * box with rewind and advance buttons on either side. The selection is then
 * restricted to an element which exists within the list.
 *
 * INPUTS
 *     list    The array which is to be stored for this option type. Note that
 *             the list is expected to be provided in sorted order.
 */
Pager.DataList = function(list) {
    this.idx   = 0;
    this.list  = list;
}

/**
 * Converts a chosen list element into the string which is to be displayed in
 * the input box.
 *
 * This may need to be overridden for a particular instance in order to better
 * format the output string.
 */
Pager.DataList.prototype.toString = function() {
    return this.list[this.idx].toString();
}

/**
 * Converts from a string input to the correct type for comparison operations
 * in DataList.skipto(). Default implementation converts to a string.
 *
 * Often will be overridden to coerse types, for example when a number is
 * selected in the pager and must be compared to a list of integers.
 */
Pager.DataList.prototype.fromString = function(valstr) {
    return valstr.toString();
}

/**
 * Increments the internal pointer into the array, keeping in mind the bounds
 * of the list.
 *
 * Typically should not be overridden.
 */
Pager.DataList.prototype.advance = function() {
    if (this.idx < this.list.length - 1)
        this.idx++;
}

/**
 * Decrements the internal pointer into the array, keeping in mind the bounds
 * of the list.
 *
 * Typically should not be overridden.
 */
Pager.DataList.prototype.rewind = function() {
    if (this.idx > 0)
        this.idx--;
}

/**
 * Select the given element in the list if it exists, otherwise choose the
 * nearest valid option.
 *
 * Typically should not be overridden.
 */
Pager.DataList.prototype.skipto = function(value) {
    this.idx = this.binsearch(this.fromString(value));
}

/**
 * Binary search to get nearest element in a list. This is used for pager
 * options which are taken from an array rather than a short list of options.
 *
 * INPUTS
 *     search    Element to search for in list.
 *
 *     lbound    The lower inclusive bound index for where to start the search.
 *               Defaults to 0.
 *
 *     ubound    The upper exclusive bound index for where to start the search.
 *               Defaults to list.length.
 *
 * RETURNS
 *     The index into list of the matching element, if found, otherwise the
 *     index of the location where the element should be inserted to maintain
 *     a sorted list.
 */
Pager.DataList.prototype.binsearch = function(search, lbound, ubound) {
    // Set default values if needed
    lbound = (typeof lbound!=='undefined' ? lbound : 0);
    ubound = (typeof ubound!=='undefined' ? ubound : this.list.length);

    // Bounds are left inclusive, right exclusive
    // (... | 0) is trick to effectively do integer division
    var middle = lbound + ((ubound-lbound)/2 | 0);

    while (lbound < ubound)
    {
        // If we matched the element, then just return the location
        if (this.list[middle] == search)
            return middle;

        // If in lower half, update ubound
        if (search < this.list[middle])
        {
            ubound = middle - 1;
        }
        // Only remaining case is in upper half, so update lbound
        else
        {
            lbound = middle + 1;
        }

        // Recalculate the middle
        middle = lbound + ((ubound-lbound)/2 | 0);
    }

    // Make sure middle doesn't end up being the exclusive ubound which can
    // end up off the end of the array
    if (middle >= this.list.length)
        middle = this.list.length - 1;

    // If we got here, just return the closest match
    return middle;
}

/**
 * Provide a default pager
 */
var pager = new Pager();
