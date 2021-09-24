str = 'https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F14N9Yjxbg2mUk8FPAFFJTTgyAY8cv9iu7'

import urllib.parse
print(urllib.parse.unquote(str))