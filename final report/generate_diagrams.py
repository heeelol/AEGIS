import graphviz

def generate_electrical_architecture():
    dot = graphviz.Digraph('Electrical_Architecture', format='png')
    dot.attr(rankdir='LR', nodesep='0.8', ranksep='1.2', fontname='Arial')
    
    # Node styles
    dot.attr('node', shape='none', fontname='Arial', margin='0')
    dot.attr('edge', fontname='Arial', fontsize='10', color='#555555')

    # Define nodes with HTML-like labels and emojis for hardware symbols
    dot.node('Camera', '''<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="10" BGCOLOR="#E8F5E9">
        <TR><TD><FONT POINT-SIZE="24">📷</FONT></TD></TR>
        <TR><TD><B>Logitech C270</B><BR/>(Vision Input)</TD></TR>
    </TABLE>>''')
    
    dot.node('Jetson', '''<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="10" BGCOLOR="#E3F2FD">
        <TR><TD><FONT POINT-SIZE="24">🖥️</FONT></TD></TR>
        <TR><TD><B>NVIDIA Jetson</B><BR/>(Core Processing Unit)</TD></TR>
    </TABLE>>''')
    
    dot.node('Display', '''<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="10" BGCOLOR="#FCE4EC">
        <TR><TD><FONT POINT-SIZE="24">📺</FONT></TD></TR>
        <TR><TD><B>LCD Monitor</B><BR/>(HMI / UI Display)</TD></TR>
    </TABLE>>''')
    
    dot.node('ESP32', '''<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="10" BGCOLOR="#FFF3E0">
        <TR><TD><FONT POINT-SIZE="24">🎛️</FONT></TD></TR>
        <TR><TD><B>ESP32 / MCU</B><BR/>(Sensor Hub)</TD></TR>
    </TABLE>>''')
    
    dot.node('LoadCells', '''<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="10" BGCOLOR="#F3E5F5">
        <TR><TD><FONT POINT-SIZE="24">⚖️</FONT></TD></TR>
        <TR><TD><B>HX711 &amp; Load Cells</B><BR/>(Force Measurement)</TD></TR>
    </TABLE>>''')
    
    dot.node('Buzzer', '''<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="10" BGCOLOR="#FFEBEE">
        <TR><TD><FONT POINT-SIZE="24">🔔</FONT></TD></TR>
        <TR><TD><B>Buzzer</B><BR/>(Audio Alert)</TD></TR>
    </TABLE>>''')

    # Connections with explicit I/O labels
    dot.edge('Camera', 'Jetson', label=' USB 2.0 \n(Video Stream in)')
    dot.edge('Jetson', 'Display', label=' HDMI \n(Video Out)')
    
    dot.edge('LoadCells', 'ESP32', label=' I2C / SPI \n(Raw Weight in)')
    dot.edge('ESP32', 'Jetson', label=' Serial / USB \n(Processed Weight in)')
    
    dot.edge('Jetson', 'Buzzer', label=' GPIO \n(PWM Alert Out)')

    # Render diagram
    dot.render('images/electrical_architecture', cleanup=True)
    print("Electrical architecture diagram generated at images/electrical_architecture.png")

def generate_flowchart():
    dot = graphviz.Digraph('System_Flowchart', format='png')
    dot.attr(rankdir='TB', fontname='Arial', nodesep='0.6', ranksep='0.6')
    
    # Global node and edge styling
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#f8f9fa', fontname='Arial', width='2')
    dot.attr('edge', fontname='Arial', fontsize='10', color='#333333')

    # Define nodes
    dot.node('Start', 'System Initialization', shape='oval', fillcolor='#e2e3e5')
    
    # Parallel processes
    with dot.subgraph(name='cluster_processes') as c:
        c.attr(label='Concurrent Processes', style='dashed', color='blue')
        c.node('CV', 'CV Thread (MediaPipe)\\nDetect Hands & Joints', fillcolor='#e3f2fd')
        c.node('LoadCell', 'Sensor Thread (ESP32)\\nPoll Load Cells', fillcolor='#fff3e0')
        c.node('HMI', 'HMI Server (FastAPI)\\nServe UI & Data', fillcolor='#fce4ec')
    
    dot.node('UpdateState', 'Update Shared State\\n(Thread-Safe)', fillcolor='#e8f5e9')
    
    # Decisions & Actions
    dot.node('CheckDanger', 'Is Hand in Danger Zone\\nAND Machine Active?', shape='diamond', fillcolor='#fff9c4')
    dot.node('TriggerAlert', 'Trigger Buzzer\\n& UI Alert', fillcolor='#ffebee')
    dot.node('Safe', 'Normal Operation\\n(No Action)', fillcolor='#f1f8e9')
    
    # Edges
    dot.edge('Start', 'CV')
    dot.edge('Start', 'LoadCell')
    dot.edge('Start', 'HMI')
    
    dot.edge('CV', 'UpdateState')
    dot.edge('LoadCell', 'UpdateState')
    
    dot.edge('UpdateState', 'CheckDanger')
    dot.edge('CheckDanger', 'TriggerAlert', label='Yes', color='red', fontcolor='red')
    dot.edge('CheckDanger', 'Safe', label='No', color='green', fontcolor='green')
    
    # HMI reads from state independently
    dot.edge('UpdateState', 'HMI', style='dashed', label=' Reads State', constraint='false')

    # Render
    dot.render('images/system_flowchart', cleanup=True)
    print("System flowchart generated at images/system_flowchart.png")

if __name__ == '__main__':
    generate_electrical_architecture()
    generate_flowchart()
