using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace DataMarker
{

    public partial class Form1 : Form
    {
        public string data_path;
        public string list_path;
        public int current_index;
        public List<FileInfo> file_list;
        public Dictionary<string,bool> recorded_labels;

        public Form1()
        {
            InitializeComponent();
            this.data_path = "";
        }
        private void init_status()
        {
            this.recorded_labels = new Dictionary<string, bool>();
            this.list_path = this.data_path + "\\valid_file.txt";
            FileStream FS = new FileStream(this.list_path, FileMode.OpenOrCreate);
            StreamReader SR = new StreamReader(FS);
            while (!SR.EndOfStream)
            {
                string[] content = SR.ReadLine().Split('\t');
                if (content.Length == 2)
                {
                    this.recorded_labels.Add(content[0], content[1].Equals("1"));
                }
            }
            SR.Close();
            FS.Close();

            DirectoryInfo DI = new DirectoryInfo(this.data_path);
            this.file_list = DI.GetFiles().ToList<FileInfo>();
            for(int i = 0; i < this.file_list.Count; i++)
            {
                if (this.file_list[i].Name == "valid_file.txt")
                {
                    this.file_list.RemoveAt(i);
                    break;
                }
            }
            this.current_index = 0;
            string initial_pic = this.file_list[this.current_index].FullName;
            if (initial_pic.Replace(".jpg",".JPG").EndsWith(".JPG")|| initial_pic.Replace(".png", ".PNG").EndsWith(".PNG"))
            {
                this.ImageBox.Load(this.file_list[this.current_index].FullName);
                this.refresh_label();
            }
            else
            {
                MessageBox.Show("所选目录存在非图片文件，请重新选择。");
                this.file_list = null;
            }
        }

        private void refresh_label()
        {
            string current_img = this.file_list[this.current_index].Name;
            if (this.recorded_labels.ContainsKey(current_img))
            {
                this.mark_pass.Enabled = !this.recorded_labels[current_img];
                this.mark_fail.Enabled = this.recorded_labels[current_img];
            }
            else
            {
                this.mark_pass.Enabled = true;
                this.mark_fail.Enabled = true;
            }
        }

        private void load_list_Click(object sender, EventArgs e)
        {
            FolderBrowserDialog OFD = new FolderBrowserDialog();
            OFD.Description = "选择数据及列表所在目录";
            if (OFD.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    this.data_path = OFD.SelectedPath;
                    this.init_status();
                }
                catch (Exception) {
                    MessageBox.Show("所选目录无效且引发异常，请重新选择。");
                }
            }
        }

        private void prev_img_Click(object sender, EventArgs e)
        {
            if (this.file_list != null)
            {
                this.current_index--;
                if (this.current_index < 0)
                {
                    this.current_index = 0;
                }
                string next_file = this.file_list[this.current_index].FullName;
                if (next_file.Replace(".jpg", ".JPG").EndsWith(".JPG") || next_file.Replace(".png", ".PNG").EndsWith(".PNG"))
                {
                    this.ImageBox.Load(this.file_list[this.current_index].FullName);
                    this.refresh_label();
                }
                else
                {
                    MessageBox.Show("所选目录存在非图片文件，请重新选择。");
                    this.current_index = 0;
                    Graphics g = this.ImageBox.CreateGraphics();
                    g.Clear(Color.WhiteSmoke);
                    this.file_list = null;
                }
            }
            else
            {
                MessageBox.Show("未指定数据所在目录，请先选择图片数据文件夹。");
            }
        }

        private void next_img_Click(object sender, EventArgs e)
        {
            if (this.file_list != null)
            {
                this.current_index++;
                if (this.current_index >= this.file_list.Count)
                {
                    this.current_index = this.file_list.Count - 1;
                }
                string next_file = this.file_list[this.current_index].FullName;
                if (next_file.Replace(".jpg", ".JPG").EndsWith(".JPG") || next_file.Replace(".png", ".PNG").EndsWith(".PNG"))
                {
                    this.ImageBox.Load(this.file_list[this.current_index].FullName);
                    this.refresh_label();
                }
                else
                {
                    MessageBox.Show("所选目录存在非图片文件，请重新选择。");
                    this.current_index = 0;
                    Graphics g = this.ImageBox.CreateGraphics();
                    g.Clear(Color.WhiteSmoke);
                    this.file_list = null;
                }
            }
            else
            {
                MessageBox.Show("未指定数据所在目录，请先选择图片数据文件夹。");
            }
        }

        private void mark_pass_Click(object sender, EventArgs e)
        {
            if (this.file_list != null)
            {
                string current_file = this.file_list[this.current_index].Name;
                if (this.recorded_labels.ContainsKey(current_file))
                {
                    recorded_labels[current_file] = true;
                }
                else
                {
                    recorded_labels.Add(current_file, true);
                }
                this.refresh_label();
            }
            else
            {
                MessageBox.Show("未指定数据所在目录，请先选择图片数据文件夹。");
            }

        }

        private void mark_fail_Click(object sender, EventArgs e)
        {
            if (this.file_list != null)
            {
                string current_file = this.file_list[this.current_index].Name;
                if (this.recorded_labels.ContainsKey(current_file))
                {
                    recorded_labels[current_file] = false;
                }
                else
                {
                    recorded_labels.Add(current_file, false);
                }
                this.refresh_label();

            }
            else
            {
                MessageBox.Show("未指定数据所在目录，请先选择图片数据文件夹。");
            }
        }

        private void save_list_Click(object sender, EventArgs e)
        {
            if (this.list_path != null)
            {
                FileStream FS = new FileStream(this.list_path, FileMode.OpenOrCreate);
                StreamWriter SW = new StreamWriter(FS);
                List<string> file_out = this.recorded_labels.Keys.ToList<string>();
                foreach (string file in file_out)
                {
                    SW.WriteLine(file + "\t" + Convert.ToInt32(this.recorded_labels[file]).ToString());
                }
                SW.Close();
                FS.Close();

            }
            else
            {
                MessageBox.Show("未指定数据所在目录，请先选择图片数据文件夹。");
            }
        }
    }
}
