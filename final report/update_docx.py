import docx

def update_chapter_3(doc_path, out_path):
    doc = docx.Document(doc_path)
    
    ch3_idx = -1
    ch4_idx = -1
    for i, p in enumerate(doc.paragraphs):
        text = p.text.strip().lower()
        if 'chapter 3: prototyping and iteration' in text or 'chapter 3: prototype development' in text:
            if ch3_idx == -1:
                ch3_idx = i
        if 'chapter 4: testing' in text or 'chapter 4 | testing' in text or 'chapter 4:' in text:
            if ch3_idx != -1 and ch4_idx == -1:
                ch4_idx = i
                break
            
    if ch3_idx != -1 and ch4_idx != -1:
        print(f"Found Ch3 at {ch3_idx}, Ch4 at {ch4_idx} in {doc_path}")
        
        # Delete paragraphs between Ch3 and Ch4
        for _ in range(ch4_idx - ch3_idx - 1):
            p = doc.paragraphs[ch3_idx + 1]
            p._element.getparent().remove(p._element)
            
        # Rename Chapter 3 Heading
        doc.paragraphs[ch3_idx].text = "Chapter 3: Prototype Development and Iteration"
            
        # Read the new markdown content
        with open("/home/jw/.gemini/antigravity-cli/brain/93288b53-c236-4705-81d9-bfd87dcfe5df/chapter_3_rewrite.md", "r") as f:
            content = f.read()
            
        # Insert new paragraphs
        ch4_p = doc.paragraphs[ch3_idx + 1] # this is now Ch4 because we deleted the middle
        
        # Skip the first line since we already set the header
        lines = content.split('\n\n')
        for line in lines[1:]:
            if line.strip():
                new_p = ch4_p.insert_paragraph_before(line.strip())
                if line.startswith('## '):
                    new_p.style = 'Heading 2'
                    new_p.text = line.replace('## ', '')
                elif line.startswith('### '):
                    new_p.style = 'Heading 3'
                    new_p.text = line.replace('### ', '')
                else:
                    new_p.style = 'Normal'
        
        doc.save(out_path)
        print("Updated doc saved to", out_path)
    else:
        print(f"Could not find Chapter 3 or 4 bounds in {doc_path}. Found Ch3: {ch3_idx}, Ch4: {ch4_idx}")

update_chapter_3("draft2.docx", "draft2.docx")
update_chapter_3("IS305 - Final Report_draft2_filled.docx", "IS305 - Final Report_draft2_filled.docx")
