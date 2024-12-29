from pathlib import Path
from ray.data import from_items
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataChunker:
    def __init__(self, DOCS_DIR, chunk_size, chunk_overlap, URL):
        self.DOCS_DIR = Path(DOCS_DIR)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.URL = URL
    

    def _load_data(self):
        print(f"Loading data from {self.DOCS_DIR}...", end="")
        self.ds = from_items([{"path": path} for path in self.DOCS_DIR.rglob("*.html") if not path.is_dir()])
        print("Done")
        print(f"{self.ds.count()} documents loaded")

    
    def _extract_text_from_section(self, section):
        texts = []

        for elem in section.children:
            if isinstance(elem, NavigableString):
                if elem.strip():
                    texts.append(elem.strip())
            elif elem.name == "section":
                continue
            else:
                texts.append(elem.get_text().strip())
                
        return "\n".join(texts)


    def _path_to_uri(self, path, scheme="https://"):
        return scheme + self.URL + str(path).split(self.URL)[-1]


    def _extract_sections(self, record):
        with open(record["path"], "r", encoding="utf-8") as html_file:
            soup = BeautifulSoup(html_file, "html.parser")

        sections = soup.find_all("section")
        section_list = []

        print(f"Extracting sections from {record['path']}...", end="")
        for section in sections:
            section_id = section.get("id")
            section_text = self._extract_text_from_section(section)

            if section_id:
                URI = self._path_to_uri(path=record["path"])
                section_list.append({"source": f"{URI}#{section_id}", "text": section_text})
        print("Done")

        return section_list


    def _chunk_section(self, section):
        print(f"Chunking section {section['source']}...", end="")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len)
        chunks = text_splitter.create_documents(
            texts=[section["text"]], 
            metadatas=[{"source": section["source"]}])
        print("Done")

        return [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]
    

    def __call__(self):
        self._load_data()

        sections_ds = self.ds.flat_map(self._extract_sections)
        chunks_ds = sections_ds.flat_map(self._chunk_section)

        return chunks_ds
