import mwxml
import mwparserfromhell
import argparse
import re
from mwparserfromhell.wikicode import Heading


HEADINGS = r"==[^=].*[^=]==\n"
EDGE_CASE = ("file:", "image:", "category:")
ARTICLE_MIN_WORDS = 50
REMOVED_SECTIONS = (
    "reference",
    "see also",
    "further reading",
    "sources",
    "citation",
    "external links"
)

"""Extensions passed on to the image decoder."""
IMAGE_EXTENSIONS = (
    "blp",
    "bmp",
    "dib",
    "bufr",
    "cur",
    "pcx",
    "dcx",
    "dds",
    "ps",
    "eps",
    "fit",
    "fits",
    "fli",
    "flc",
    "ftc",
    "ftu",
    "gbr",
    "gif",
    "grib",
    "h5",
    "hdf",
    "png",
    "apng",
    "jp2",
    "j2k",
    "jpc",
    "jpf",
    "jpx",
    "j2c",
    "icns",
    "ico",
    "im",
    "iim",
    "tif",
    "tiff",
    "jfif",
    "jpe",
    "jpg",
    "jpeg",
    "mpg",
    "mpeg",
    "msp",
    "pcd",
    "pxr",
    "pbm",
    "pgm",
    "ppm",
    "pnm",
    "psd",
    "bw",
    "rgb",
    "rgba",
    "sgi",
    "ras",
    "tga",
    "icb",
    "vda",
    "vst",
    "webp",
    "wmf",
    "emf",
    "xbm",
    "xpm",
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    args = parser.parse_args()
    
    def page_info(dump, path):
        for page in dump:
            yield page
    
    flag = True
    got_info = False
    for p_idx, page in enumerate(mwxml.map(page_info, [args.input])):
        if page.namespace != 0:
            continue

        print(f"=" * 10)
        print(f" Proces article {page.title} ")
        print(f"=" * 10)

        # assume there is only 1 revision
        revision = next(page)
        
        if str(revision.text).startswith('#REDIRECT'):
            print("Redirect detected, skip this article")
            continue

        parsed = mwparserfromhell.parse(revision.text)
        
        # info images belongs to intro (leading) section
        info_images = set()
        # find infobox and grab images
        for template in parsed.filter_templates():
            if template.name.strip().lower().startswith('infobox'):
                for param in template.params:
                    value = param.value.strip().lower()
                    extension = re.sub(r".*[.]", "", value)
                    if extension in IMAGE_EXTENSIONS:
                        info_images.add(param.value.strip().replace(' ', '_'))

        # get the "introduction" section by include_lead
        for s_idx, section in enumerate(parsed.get_sections(include_lead=True, flat=True)):

            if s_idx == 0:
                title = 'Intro'
                level = 2
            else:
                heading = section.filter_headings()[0]
                title = heading.title
                tags = title.filter_tags(matches=lambda tag: tag.tag == "span")
                for tag in tags:
                    title.remove(tag)
                title = title.strip_code(normalize=True, collapse=True).strip()
                level = heading.level
                # remove section title from context
                section.remove(heading)
            
            if title.lower().startswith(REMOVED_SECTIONS):
                continue


            # Cleaining ...
            # Process edge cases of mwparser:
            for link in section.filter_wikilinks():
                if link.title.strip().lower().startswith(EDGE_CASE):
                    section.remove(link)
            # remove tags
            for tag in section.filter_tags(matches=lambda tag: tag.tag == "ref"):
                section.remove(tag)
            
            section_contents = section.strip_code(normalize=True, collapse=True)
            section_contents = re.sub(r"\n\n+", "\n", section_contents).strip()
            
            # Assume space seperated ...
            if len(section_contents.split()) < ARTICLE_MIN_WORDS:
                continue
            
            print('='*10)
            print(f"Section: {title}, Level: {level}")
            print(len(section_contents.split()))

            if flag:
                continue
            else:
                break
        
        if p_idx > 5000:
            flag = False
        
        if flag:
            continue
        else: 
            break
        
        