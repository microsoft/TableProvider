import numpy as np
import pandas as pd
from tabulate import tabulate


def to_xml(df):
    def row_to_xml(row):
        xml = ['<item>']
        for i, col_name in enumerate(row.index):
            xml.append('  <field name="{0}">{1}</field>'.format(col_name, row.iloc[i]))
        xml.append('</item>')
        return '\n'.join(xml)

    res = '\n'.join(df.apply(row_to_xml, axis=1))
    return res


class StructuredDataLinearize:
    """Expects the structured data with the following format:

        structured_data_dict = {
                "title": "",
                "context": example["context"] + example["passage"],
                "table": {
                    "header": example['table']['header'],
                    "rows": example['table']['rows'],
                    "caption": ""
                }
            }
    """

    def __init__(self):
        self.end_prompt = "The answer is "
        pd.DataFrame.to_xml = to_xml

    def retrieve_linear_function(self, func, use_structure_mark, add_grammar, change_order, structured_data_dict):
        self.structured_data_dict = structured_data_dict
        self.use_structure_mark = use_structure_mark
        self.add_grammar = add_grammar  # add grammar description of the format
        self.change_order = change_order  # if true, the table will change from row-major to column major
        dict = {
            "markdown": self.linearize_markdown,
            "markdown_grid": self.linearize_markdown_grid,
            "xml": self.linearize_xml,
            "html": self.linearize_html,
            "json": self.linearize_json,
            "latex": self.linearize_latex,
            "nl_sep": self.linear_nl_sep,
        }
        return dict[func]()

    def linearize_markdown(self):
        if self.use_structure_mark:
            additional_knowledge = "<title>\n" + "".join(
                self.structured_data_dict["title"]) + "\n" + "<context>\n" + "".join(
                self.structured_data_dict["context"]) + "\n" + "<caption>\n" + "".join(
                "".join(self.structured_data_dict["table"]["caption"])) + "\n"
        else:
            additional_knowledge = "".join(self.structured_data_dict["title"]) + "\n" + "".join(
                self.structured_data_dict["context"]) + "\n" + "".join(
                self.structured_data_dict["table"]["caption"]) + "\n"
        if self.change_order:
            structured_data = pd.DataFrame(np.array(self.structured_data_dict['table']['rows']).T,
                                           columns=self.structured_data_dict['table']['header'])
            structured_data_markdown = tabulate(structured_data, tablefmt='pipe', showindex=True)
        else:
            structured_data = pd.DataFrame(self.structured_data_dict['table']['rows'], columns=self.structured_data_dict['table']['header'])
            structured_data_markdown = tabulate(structured_data, tablefmt='pipe', showindex=True)

        if self.add_grammar:
            grammar = "<Markdown grammar>\n To add a table, use three or more hyphens (---) to create each column’s header, and use pipes (|) to separate each column, every cell is separated by pipe \n"
            return additional_knowledge + grammar + structured_data_markdown + "\n" + self.end_prompt
        else:
            return additional_knowledge + structured_data_markdown + "\n" + self.end_prompt

    def linearize_markdown_grid(self):
        if self.use_structure_mark:
            additional_knowledge = "<title>\n" + "".join(
                self.structured_data_dict["title"]) + "\n" + "<context>\n" + "".join(
                self.structured_data_dict["context"]) + "\n" + "<caption>\n" + "".join(
                self.structured_data_dict["table"]["caption"]) + "\n"
        else:
            additional_knowledge = "".join(self.structured_data_dict["title"]) + "\n" + "".join(
                self.structured_data_dict["context"]) + "\n" + "".join(
                self.structured_data_dict["table"]["caption"]) + "\n"
        if self.change_order:
            structured_data = pd.DataFrame(np.array(self.structured_data_dict['table']['rows']).T,
                                           index=self.structured_data_dict['table']['header'])
            structured_data_markdown = tabulate(structured_data, tablefmt='pipe', showindex=True)
        else:
            structured_data = pd.DataFrame(self.structured_data_dict['table']['rows'])
            structured_data_markdown = tabulate(structured_data, headers=self.structured_data_dict['table']['header'],
                                                tablefmt='grid', showindex=True)
        if self.add_grammar:
            grammar = "<Markdown grammar>\n To add a table, use three or more hyphens (---) to create each column’s header, and use pipes (|) to separate each column, every cell is separated by pipe \n" \
                      "Grid is like tables formatted by Emacs' table.el package. It corresponds to grid_tables in Pandoc Markdown extensions\n"
            return additional_knowledge + grammar + structured_data_markdown + "\n" + self.end_prompt
        else:
            return additional_knowledge + structured_data_markdown + "\n" + self.end_prompt

    def linearize_xml(self):
        if self.use_structure_mark:
            additional_knowledge = "<title>\n" + "".join(
                self.structured_data_dict["title"]) + "\n" + "<context>\n" + "".join(
                self.structured_data_dict["context"]) + "\n" + "<caption>\n" + "".join(
                self.structured_data_dict["table"]["caption"]) + "\n"
        else:
            additional_knowledge = "".join(self.structured_data_dict["title"]) + "\n" + "".join(
                self.structured_data_dict["context"]) + "\n" + "".join(
                self.structured_data_dict["table"]["caption"]) + "\n"
        header = self.structured_data_dict['table']['header']
        for i in range(len(header)):
            header[i] = "_".join(header[i].split())
        if self.change_order:
            structured_data = pd.DataFrame(np.array(self.structured_data_dict['table']['rows']).T, columns=self.structured_data_dict['table']['header'])
            structured_data_xml = structured_data.to_xml()
        else:
            structured_data = pd.DataFrame(self.structured_data_dict['table']['rows'], columns=self.structured_data_dict['table']['header'])
            structured_data_xml = structured_data.to_xml()

        if self.add_grammar:
            grammar = "<XML grammar>\n <?xml version='1.0' encoding='utf-8'?>\n<data>\n  <row>\n    <index>0</index>\n    <column_1>2</<column_1>>\n  </row>\n  <row>\n    <index>1</index>\n    <column_2>4</column_2>\n  </row>\n</data>"
            return additional_knowledge + grammar + structured_data_xml + "\n" + self.end_prompt
        else:
            return additional_knowledge + structured_data_xml + "\n" + self.end_prompt

    def linearize_html(self):
        if self.use_structure_mark:
            additional_knowledge = "<title>\n" + "".join(self.structured_data_dict["title"]) \
                if self.structured_data_dict["title"] != "" else ""

            additional_knowledge += "\n" + "<context>\n" + "".join(self.structured_data_dict["context"]) \
                if self.structured_data_dict["context"] != "" else ""

            additional_knowledge += "\n" + "<caption>\n" + "".join(
                "".join(self.structured_data_dict["table"]["caption"])) \
                if self.structured_data_dict["table"]["caption"] != "" else ""
            if additional_knowledge != "":
                additional_knowledge += "\n"
        else:
            additional_knowledge = "".join(self.structured_data_dict["title"]) + "\n" + "".join(
                self.structured_data_dict["context"]) + "\n" + "".join(
                self.structured_data_dict["table"]["caption"]) + "\n"

        rows = len(self.structured_data_dict['table']['rows'])
        columns = len(self.structured_data_dict['table']['rows'][0])
        additional_knowledge += "\n" + f"The table has {rows} rows and {columns} columns \n"


        if self.change_order:
            header = False if len(self.structured_data_dict['table']['header']) == 1 and \
                              self.structured_data_dict['table']['header'][0] == "" else True
            structured_data = pd.DataFrame(np.array(self.structured_data_dict['table']['rows']).T,
                                           columns=self.structured_data_dict['table']['header'])
            structured_data_html = structured_data.to_html(header=header)
        else:
            header = False if len(self.structured_data_dict['table']['header']) == 1 and \
                              self.structured_data_dict['table']['header'][0] == "" else True
            structured_data = pd.DataFrame(self.structured_data_dict['table']['rows'],
                                           columns=self.structured_data_dict['table']['header'])
            structured_data_html = structured_data.to_html(header=header)
        if self.add_grammar:
            grammar = "<HTML grammar>\n Each table cell is defined by a <td> and a </td> tag.\n Each table row starts with a <tr> and ends with a </tr> tag.\n th stands for table header.\n"
            return additional_knowledge + grammar + structured_data_html + "\n" + self.end_prompt
        else:
            return additional_knowledge + structured_data_html + "\n" + self.end_prompt

    def linearize_json(self):
        # convert a json file to string, already have the structure mark
        if self.add_grammar:
            grammar = "<JSON grammer>\n JSON is built of a collection of name/value pairs. Each pair is key-value\n"
            return grammar + str(self.structured_data_dict)
        else:
            return str(self.structured_data_dict)

    def linearize_latex(self):
        if self.use_structure_mark:
            additional_knowledge = "<title>\n" + "".join(
                self.structured_data_dict["title"]) + "\n" + "<context>\n" + "".join(
                self.structured_data_dict["context"]) + "\n" + "<caption>\n" + "".join(
                self.structured_data_dict["table"]["caption"]) + "\n"
        else:
            additional_knowledge = "".join(self.structured_data_dict["title"]) + "\n" + "".join(
                self.structured_data_dict["context"]) + "\n" + "".join(
                self.structured_data_dict["table"]["caption"]) + "\n"
        if self.change_order:
            structured_data = pd.DataFrame(np.array(self.structured_data_dict['table']['rows']).T,
                                           columns=self.structured_data_dict['table']['header'])
            structured_data_latex = structured_data.to_latex()
        else:
            structured_data = pd.DataFrame(self.structured_data_dict['table']['rows'],
                                           columns=self.structured_data_dict['table']['header'])
            structured_data_latex = structured_data.to_latex()
        if self.add_grammar:
            grammar = "<Latex grammar>\n \begin{tabular} starts the table environment and the curly braces denote the alignment of the columns.\n |c|c|c| means that the table has three columns and each column is center-aligned.\n " \
                      "\hline creates a horizontal line.\n The text in between the & symbols is the content of the cells.\n '\\' is used to end a row.\n \end{tabular} ends the table environment.\n"
            return additional_knowledge + grammar + structured_data_latex + "\n" + self.end_prompt
        else:
            return additional_knowledge + structured_data_latex + "\n" + self.end_prompt

    def linear_nl_sep(self):
        if self.use_structure_mark:
            additional_knowledge = "<title>\n" + "".join(
                self.structured_data_dict["title"]) + "\n" + "<context>\n" + "".join(
                self.structured_data_dict["context"]) + "\n" + "<caption>\n" + "".join(
                self.structured_data_dict["table"]["caption"]) + "\n"
        else:
            additional_knowledge = "".join(self.structured_data_dict["title"]) + "\n" + "".join(
                self.structured_data_dict["context"]) + "\n" + "".join(
                self.structured_data_dict["table"]["caption"]) + "\n"
        if self.change_order:
            header = self.structured_data_dict["table"]["header"]
            reversed_table = np.array(self.structured_data_dict['table']['rows']).T
            cells = []
            for i in range(len(reversed_table)):
                cells.append(header[i] + "|".join(reversed_table[i]) + "\n")
            structured_data_nl_sep = "".join(cells)
        else:
            header = "|".join(self.structured_data_dict["table"]["header"]) + "\n"
            cells = []
            for i in range(len(self.structured_data_dict["table"]["rows"])):
                cells.append("|".join(self.structured_data_dict["table"]["rows"][i]) + "\n")
            structured_data_nl_sep = header + "".join(cells)
        if self.add_grammar:
            grammar = "<Grammar>\n Each table cell is separated by | , the column idx starts from 0, .\n"
            return additional_knowledge + grammar + structured_data_nl_sep + "\n" + self.end_prompt
        else:
            return additional_knowledge + structured_data_nl_sep + "\n" + self.end_prompt
